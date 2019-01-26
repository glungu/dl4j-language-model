package org.lungen.deeplearning.net.classifier;

import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.lungen.deeplearning.iterator.StringClassifierIterator;
import org.lungen.deeplearning.listener.EarlyStopListener;
import org.lungen.deeplearning.listener.ScorePrintListener;
import org.lungen.deeplearning.listener.UIStatsListener;
import org.lungen.deeplearning.model.ModelPersistence;
import org.lungen.deeplearning.net.NeuralNet;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import static org.lungen.deeplearning.iterator.CharactersSets.*;

/**
 * CharacterSequenceValuePredictorNet
 *
 * @author lungen.tech@gmail.com
 */
public class StringClassifierNet implements NeuralNet {

    private static final Logger log = LoggerFactory.getLogger("net.classifier");

    private ComputationGraph net;
    private EarlyStopListener earlyStopListener;
    private ScorePrintListener scorePrintListener;
    private StringClassifierIterator iteratorTrain;
    private StringClassifierIterator iteratorTest;
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
//        int tbpttSize = (Integer) params.get(PARAM_TRUNCATED_BPTT_SIZE);

        final NeuralNetConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
                .seed(7)
                .weightInit(WeightInit.XAVIER)
                .l2(l2Regularization)
//                .cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE)
                .updater(new Adam(learningRate));
//                .updater(new RmsProp(learningRate))
//                .updater(new Nesterovs(learningRate, 0.9))
//                .updater(new AdaGrad(learningRate));
//                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer);

        final int hiddenRecurrentSize = 250;
        final int hiddenDenseSize = 250;

        final ComputationGraphConfiguration.GraphBuilder graphBuilder = builder.graphBuilder()
//                .backpropType(BackpropType.TruncatedBPTT)
//                .tBPTTForwardLength(tbpttSize)
//                .tBPTTBackwardLength(tbpttSize)
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
//                .addVertex("thoughtVector",
//                        new LastTimeStepVertex("recurrentInput"),
//                        "lstm-3")
                .addLayer("thoughtVector",
                        new GlobalPoolingLayer.Builder()
                                .poolingType(PoolingType.AVG)
                                .poolingDimensions(2) // recurrent dimension
                                .collapseDimensions(true)
                                .build(), "lstm-3")
//                .addLayer("norm-1",
//                        new BatchNormalization.Builder()
//                                .nIn(hiddenRecurrentSize)
//                                .nOut(hiddenRecurrentSize)
//                                .build(), "thoughtVector")
                .addLayer("dense-1",
                        new DenseLayer.Builder()
                                .activation(Activation.LEAKYRELU)
                                .nIn(hiddenRecurrentSize)
                                .nOut(hiddenDenseSize)
                                .build(), "thoughtVector")
//                .addLayer("norm-2",
//                        new BatchNormalization.Builder()
//                                .nIn(hiddenDenseSize)
//                                .nOut(hiddenDenseSize)
//                                .build(), "dense-1")
                .addLayer("dense-2",
                        new DenseLayer.Builder()
                                .activation(Activation.LEAKYRELU)
                                .nIn(hiddenDenseSize)
                                .nOut(hiddenDenseSize)
                                .build(), "dense-1")
//                .addLayer("norm-3",
//                        new BatchNormalization.Builder()
//                                .nIn(hiddenDenseSize)
//                                .nOut(hiddenDenseSize)
//                                .build(), "dense-2")
                .addLayer("output",
                        new OutputLayer.Builder()
                                .activation(Activation.SOFTMAX)
                                .lossFunction(LossFunctions.LossFunction.MCXENT)
                                .nIn(hiddenDenseSize)
                                .nOut(iteratorTest.totalOutcomes())
                                .build(), "dense-2")
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

            while (iteratorTrain.hasNext()) {
                DataSet ds = iteratorTrain.next();
                net.fit(ds);

                if (++miniBatchNumber % checkAfterNMinibatches == 0) {

                    // evaluate
//                    iteratorTest.reset();
//                    Evaluation evaluation = net.evaluate(iteratorTest);
//                    log.info("[{}][{}] Test set evaluation. Accuracy: {}, F1: {}", i, miniBatchNumber, evaluation.accuracy(), evaluation.f1());

                    iteratorTest.reset();
                    DataSet testSet = iteratorTest.next(100);
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
        int sumCorrect = 0;
        for (int i = 0; i < output.size(0); i++) {
            INDArray resultArray = output.get(NDArrayIndex.point(i), NDArrayIndex.all());
            List<Double> resultList = Arrays.stream(resultArray.toDoubleVector()).boxed().collect(Collectors.toList());
            int result = resultArray.argMax(1).getInt(0);
            int expected = labels.get(NDArrayIndex.point(i), NDArrayIndex.all()).argMax(1).getInt(0);
            evalMsg.append(resultList).append(", ").append(result).append(" : ").append(expected).append('\n');
            if (result == expected) {
                sumCorrect++;
            }
        }
        double score = sumCorrect / (double) output.size(0);
        evalMsg.append("-----------------\n");
        evalMsg.append("Score: ").append(score).append('\n');
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
    public StringClassifierIterator iterator(Map<String, Object> params) {
        String fileTrain        = (String) params.get(PARAM_DATA_FILE);
        String fileTest         = (String) params.get(PARAM_DATA_FILE_TEST);
        int minibatchSize       = (Integer) params.get(PARAM_MINIBATCH_SIZE);

        this.iteratorTrain = new StringClassifierIterator(new File(fileTrain),
                createCharacterSet(RUSSIAN, LATIN, NUMBERS, PUNCTUATION, SPECIAL),
                2000, minibatchSize);

        this.iteratorTest = new StringClassifierIterator(new File(fileTest),
                createCharacterSet(RUSSIAN, LATIN, NUMBERS, PUNCTUATION, SPECIAL),
                2000, 20);

        return iteratorTrain;
    }

    @Override
    public Map<String, Object> defaultParams() {
        Map<String, Object> params = new HashMap<>();
        params.put(PARAM_MODEL_NAME, "bugzilla-time");
        params.put(PARAM_DATA_FILE, "C:/DATA/Projects/DataSets/Bugzilla/bugzilla.train.new.csv");
        params.put(PARAM_DATA_FILE_TEST, "C:/DATA/Projects/DataSets/Bugzilla/bugzilla.test.new.csv");
        params.put(PARAM_MINIBATCH_SIZE, 32);
        params.put(PARAM_LEARNING_RATE, 1e-4);
        params.put(PARAM_L2_REGULARIZATION, 1e-6);
        params.put(PARAM_TRUNCATED_BPTT_SIZE, 100);
        params.put(PARAM_CHECK_EACH_NUMBER_MINIBATCHES, 10);
        params.put(PARAM_STOP_AFTER_NUMBER_MINIBATCHES, -1);
        params.put(PARAM_NUMBER_EPOCHS, 5);
        params.put(PARAM_NUMBER_ITER_NO_IMPROVE_STOP, 2500);
        return params;
    }

    public static void main(String[] args) {
        StringClassifierNet net = new StringClassifierNet();
        Map<String, Object> params = net.defaultParams();
        net.init(params);
        net.train(params);
    }
}
