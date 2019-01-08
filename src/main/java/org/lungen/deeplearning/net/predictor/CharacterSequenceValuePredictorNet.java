package org.lungen.deeplearning.net.predictor;

import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.graph.rnn.LastTimeStepVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.BatchNormalization;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.lungen.deeplearning.iterator.CharacterSequenceValuePredictorIterator;
import org.lungen.deeplearning.listener.EarlyStopListener;
import org.lungen.deeplearning.listener.ScorePrintListener;
import org.lungen.deeplearning.listener.UIStatsListener;
import org.lungen.deeplearning.model.ModelPersistence;
import org.lungen.deeplearning.net.NeuralNet;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.*;
import java.util.stream.Collectors;

import static org.lungen.deeplearning.iterator.CharactersSets.*;

/**
 * CharacterSequenceValuePredictorNet
 *
 * @author lungen.tech@gmail.com
 */
public class CharacterSequenceValuePredictorNet implements NeuralNet {

    private static final Logger log = LoggerFactory.getLogger("net.predictor");

    private ComputationGraph net;
    private EarlyStopListener earlyStopListener;
    private ScorePrintListener scorePrintListener;
    private CharacterSequenceValuePredictorIterator iteratorTrain;
    protected CharacterSequenceValuePredictorIterator iteratorTest;
    private UIStatsListener statsListener;
    private String modelName;

    @Override
    public void init(Map<String, Object> params) {

        iterator(params);
        int numFeaturesRecurrent = iteratorTrain.getNumSequenceFeatures();

        this.modelName          = (String) params.get(PARAM_MODEL_NAME);
        double learningRate     = (Double) params.get(PARAM_LEARNING_RATE);
        double l2Regularization = (Double) params.get(PARAM_L2_REGULARIZATION);
        int numIterEarlyStop    = (Integer) params.get(PARAM_NUMBER_ITER_NO_IMPROVE_STOP);
        int minEpochsEarlyStop  = (Integer) params.getOrDefault(PARAM_MIN_EPOCHS_STOP, 0);
        int tbpttSize = (Integer) params.get(PARAM_TRUNCATED_BPTT_SIZE);

        final NeuralNetConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
                .seed(7)
                .weightInit(WeightInit.XAVIER)
                .l2(l2Regularization)
                .updater(new Adam(learningRate));
//                .updater(new RmsProp(learningRate))
//                .updater(new Nesterovs(learningRate, 0.99))
//                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer);

        final int hiddenRecurrentSize = 250;

        final ComputationGraphConfiguration.GraphBuilder graphBuilder = builder.graphBuilder()
                .backpropType(BackpropType.TruncatedBPTT)
                .tBPTTForwardLength(tbpttSize)
                .tBPTTBackwardLength(tbpttSize)
                .addInputs("recurrentInput")
                .setInputTypes(InputType.recurrent(numFeaturesRecurrent))
                .addLayer("lstm-1",
                        new LSTM.Builder()
                                .nIn(numFeaturesRecurrent)
                                .nOut(hiddenRecurrentSize)
                                .activation(Activation.TANH)
                                .build(), "recurrentInput")
                .addLayer("lstm-2",
                        new LSTM.Builder()
                                .nIn(hiddenRecurrentSize)
                                .nOut(hiddenRecurrentSize)
                                .activation(Activation.TANH)
                                .build(), "lstm-1")
                .addLayer("lstm-3",
                        new LSTM.Builder()
                                .nIn(hiddenRecurrentSize)
                                .nOut(hiddenRecurrentSize)
                                .activation(Activation.TANH)
                                .build(), "lstm-2")
                .addVertex("thoughtVector",
                        new LastTimeStepVertex("recurrentInput"), "lstm-3")
                .addLayer("norm-1",
                        new BatchNormalization.Builder()
                                .nIn(hiddenRecurrentSize)
                                .nOut(hiddenRecurrentSize)
                                .build(), "thoughtVector")
                .addLayer("dense-1",
                        new DenseLayer.Builder()
                                .activation(Activation.TANH)
                                .nIn(hiddenRecurrentSize)
                                .nOut(100)
                                .build(), "norm-1")
                .addLayer("norm-2",
                        new BatchNormalization.Builder()
                                .nIn(100)
                                .nOut(100)
                                .build(), "dense-1")
                .addLayer("output",
                        new OutputLayer.Builder()
                                .activation(Activation.SOFTMAX)
                                .lossFunction(LossFunctions.LossFunction.MCXENT)
                                .nIn(100)
                                .nOut(iteratorTest.totalOutcomes())
                                .build(), "norm-2")
                .setOutputs("output");

        net = new ComputationGraph(graphBuilder.build());
        earlyStopListener = new EarlyStopListener(modelName, numIterEarlyStop, minEpochsEarlyStop);
        scorePrintListener = new ScorePrintListener(1);
        statsListener = new UIStatsListener();
        net.setListeners(scorePrintListener, earlyStopListener, statsListener);
        net.init();
    }

    @Override
    public void train(Map<String, Object> params) {
        int numEpochs = (Integer) params.get(PARAM_NUMBER_EPOCHS);
        int checkAfterNMinibatches = (Integer) params.get(PARAM_CHECK_EACH_NUMBER_MINIBATCHES);
        int stopAfterNMinibatches = (Integer) params.get(PARAM_STOP_AFTER_NUMBER_MINIBATCHES);

        // Do training, and then generate and print samples from network
        int miniBatchNumber = 0;
        boolean stop = false;
        for (int i = 0; i < numEpochs; i++) {
            scorePrintListener.setEpoch(i);
            earlyStopListener.setEpoch(i);

            log.info("[{}] Epoch started", i);
            String str = "Test set evaluation at epoch %d: Accuracy = %.2f, F1 = %.2f";

            while (iteratorTrain.hasNext()) {
                DataSet ds = iteratorTrain.next();
                net.fit(ds);

                if (++miniBatchNumber % checkAfterNMinibatches == 0) {

                    // evaluate
                    iteratorTest.reset();

                    Evaluation evaluation = net.evaluate(iteratorTest);
                    log.info("[{}][{}] Test set evaluation. Accuracy: {}, F1: {}", i, miniBatchNumber, evaluation.accuracy(), evaluation.f1());
//                    testOutputAndScore(ds);
                    iteratorTest.reset();
                    DataSet testSet = iteratorTest.next(20);
                    testOutputAndScore(testSet);
                }
                if (stopAfterNMinibatches > 0 && miniBatchNumber >= stopAfterNMinibatches) {
                    stop = true;
                    break;
                }
                if (earlyStopListener.isStopRecommended()) {
                    earlyStopListener.writeBestModel();
                    stop = true;
                    break;
                }

            }
            if (stop) {
                break;
            }
            // Reset iterator for another epoch
            iteratorTrain.reset();
            log.info("[{}] Epoch completed", i);
        }

        statsListener.close();
        log.info("Training complete!");

        ModelPersistence.save(modelName, net);
    }

    private void testOutputAndScore(DataSet testSet) {
        net.rnnClearPreviousState();
        INDArray output = net.output(false,
                new INDArray[] {testSet.getFeatures()},
                new INDArray[] {testSet.getFeaturesMaskArray()})[0];
        INDArray labels = testSet.getLabels();
        StringBuilder evalMsg = new StringBuilder("--- Evaluation ---\n");
        for (int i = 0; i < output.size(0); i++) {
            INDArray resultArray = output.get(NDArrayIndex.point(i), NDArrayIndex.all());
            List<Double> resultList = Arrays.stream(resultArray.toDoubleVector()).boxed().collect(Collectors.toList());
            int result = resultArray.argMax(1).getInt(0);
            int expected = labels.get(NDArrayIndex.point(i), NDArrayIndex.all()).argMax(1).getInt(0);
            evalMsg.append(resultList).append(", ").append(result).append(" : ").append(expected).append('\n');
        }
        evalMsg.append("-----------------\n");
//        net.rnnClearPreviousState();
//        net.clearLayersStates();
//        double score = net.score(testSet, false);
//        evalMsg.append("Score current: ").append(net.score()).append(", test set: ").append(score).append('\n');
//        evalMsg.append("-----------------");
        log.info(evalMsg.toString());

        StringBuilder gradSummary = new StringBuilder("--- Gradients ---\n");
        net.gradient().gradientForVariable().forEach((var, grad) -> {
            Number min = grad.aminNumber();
            Number max = grad.amaxNumber();
            Number mean = grad.ameanNumber();
            int order = (int) Math.log10(min.doubleValue());
            gradSummary.append(var).append(": ").append(min).append(",").append(mean).append(",").append(max).append(", order: ").append(order).append('\n');
        });
        gradSummary.append("-----------------");
        log.info(gradSummary.toString());


//        Evaluation eval = new Evaluation(6);
//        eval.eval(labels, output);
        net.rnnClearPreviousState();
        net.clearLayersStates();
    }

    @Override
    public double getBestScore() {
        return earlyStopListener.getBestScore();
    }

    @Override
    public CharacterSequenceValuePredictorIterator iterator(Map<String, Object> params) {
        String fileTrain        = (String) params.get(PARAM_DATA_FILE);
        String fileTest         = (String) params.get(PARAM_DATA_FILE_TEST);
        int minibatchSize       = (Integer) params.get(PARAM_MINIBATCH_SIZE);

        this.iteratorTrain = new CharacterSequenceValuePredictorIterator(new File(fileTrain),
                createCharacterSet(RUSSIAN, LATIN, PUNCTUATION, SPECIAL),
                2000, minibatchSize);

        this.iteratorTest = new CharacterSequenceValuePredictorIterator(new File(fileTest),
                createCharacterSet(RUSSIAN, LATIN, PUNCTUATION, SPECIAL),
                2000, 100);

        return iteratorTrain;
    }

    @Override
    public Map<String, Object> defaultParams() {
        Map<String, Object> params = new HashMap<>();
        params.put(PARAM_MODEL_NAME, "wiktionary");
        params.put(PARAM_DATA_FILE, "C:/DATA/Projects/DataSets/Bugzilla/bugzilla.train.csv");
        params.put(PARAM_DATA_FILE_TEST, "C:/DATA/Projects/DataSets/Bugzilla/bugzilla.test.csv");
        params.put(PARAM_MINIBATCH_SIZE, 64);
        params.put(PARAM_LEARNING_RATE, 2e-4);
        params.put(PARAM_L2_REGULARIZATION, 1e-4);
        params.put(PARAM_TRUNCATED_BPTT_SIZE, 50);
        params.put(PARAM_CHECK_EACH_NUMBER_MINIBATCHES, 10);
        params.put(PARAM_STOP_AFTER_NUMBER_MINIBATCHES, -1);
        params.put(PARAM_NUMBER_EPOCHS, 3);
        params.put(PARAM_NUMBER_ITER_NO_IMPROVE_STOP, 20000);
        return params;
    }

    public static void main(String[] args) {
        CharacterSequenceValuePredictorNet net = new CharacterSequenceValuePredictorNet();
        Map<String, Object> params = net.defaultParams();
        net.init(params);
        net.train(params);
    }
}
