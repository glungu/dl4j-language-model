package org.lungen.deeplearning.net.classifier;

import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.lungen.deeplearning.iterator.CharacterSequenceClassifierIterator;
import org.lungen.deeplearning.listener.EarlyStopListener;
import org.lungen.deeplearning.listener.ScorePrintListener;
import org.lungen.deeplearning.listener.UIStatsListener;
import org.lungen.deeplearning.model.ModelPersistence;
import org.lungen.deeplearning.net.NeuralNet;
import org.lungen.deeplearning.net.autoencoder.MultivariatePredictorNet;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

import static org.lungen.deeplearning.iterator.CharactersSets.RUSSIAN;
import static org.lungen.deeplearning.iterator.CharactersSets.createCharacterSet;

public class CharacterSequenceClassifierNet implements NeuralNet {

    private static final Logger log = LoggerFactory.getLogger("net.classifier");

    private ComputationGraph net;
    private EarlyStopListener earlyStopListener;
    private ScorePrintListener scorePrintListener;
    private CharacterSequenceClassifierIterator iteratorTrain;
    private CharacterSequenceClassifierIterator iteratorTest;
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
        int numLabelClasses     = (Integer) params.get(PARAM_NUMBER_OUTPUT_CLASSES);

        final NeuralNetConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
//                .updater(new RmsProp(learningRate))
                .seed(7)
                .weightInit(WeightInit.XAVIER)
                .l2(l2Regularization)
                .updater(new Adam(learningRate))
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer);

        final int hiddenRecurrentSize = 250;

        final ComputationGraphConfiguration.GraphBuilder graphBuilder = builder.graphBuilder()
                .backpropType(BackpropType.Standard)
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
                .layer("output",
                        new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                                .activation(Activation.SOFTMAX)
                                .nIn(hiddenRecurrentSize)
                                .nOut(numLabelClasses)
                                .build(), "lstm-2")
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
                    Evaluation evaluation = net.evaluate(iteratorTest);
                    log.info("[{}][{}] Test set evaluation. Accuracy: {}, F1: {}", i, miniBatchNumber, evaluation.accuracy(), evaluation.f1());
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
            iteratorTest.reset();
            log.info("[{}] Epoch completed", i);
        }

        statsListener.close();
        log.info("Training complete!");

        ModelPersistence.save(modelName, net);
    }

    @Override
    public double getBestScore() {
        return 0;
    }

    @Override
    public CharacterSequenceClassifierIterator iterator(Map<String, Object> params) {
        String fileTrain        = (String) params.get(PARAM_DATA_FILE);
        String fileTest         = (String) params.get(PARAM_DATA_FILE_TEST);
        int minibatchSize       = (Integer) params.get(PARAM_MINIBATCH_SIZE);
        int numOutputClasses    = (Integer) params.get(PARAM_NUMBER_OUTPUT_CLASSES);

        this.iteratorTrain = new CharacterSequenceClassifierIterator(new File(fileTrain),
                createCharacterSet(RUSSIAN, Collections.singletonList('-')),
                numOutputClasses, minibatchSize);
        this.iteratorTest = new CharacterSequenceClassifierIterator(new File(fileTest),
                createCharacterSet(RUSSIAN, Collections.singletonList('-')),
                numOutputClasses, 1500);

        return iteratorTrain;
    }

    @Override
    public Map<String, Object> defaultParams() {
        Map<String, Object> params = new HashMap<>();
        params.put(PARAM_MODEL_NAME, "bugzilla");
        params.put(PARAM_DATA_FILE, "C:/DATA/Projects/DataSets/RU_Wiktionary/words.train.csv");
        params.put(PARAM_DATA_FILE_TEST, "C:/DATA/Projects/DataSets/RU_Wiktionary/words.test.csv");
        params.put(PARAM_NUMBER_OUTPUT_CLASSES, 3);
        params.put(PARAM_MINIBATCH_SIZE, 32);
        params.put(PARAM_SEQUENCE_LENGTH, 1000);
        params.put(PARAM_LEARNING_RATE, 1e-3);
        params.put(PARAM_L2_REGULARIZATION, 1e-3);
        params.put(PARAM_TRUNCATED_BPTT_SIZE, 50);
        params.put(PARAM_CHECK_EACH_NUMBER_MINIBATCHES, 1);
        params.put(PARAM_STOP_AFTER_NUMBER_MINIBATCHES, -1);
        params.put(PARAM_NUMBER_EPOCHS, 200);
        params.put(PARAM_NUMBER_ITER_NO_IMPROVE_STOP, 20000);
        return params;
    }

    public static void main(String[] args) {
        MultivariatePredictorNet net = new MultivariatePredictorNet();
        Map<String, Object> params = net.defaultParams();
        net.init(params);
        net.train(params);
    }
}

