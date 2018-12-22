package org.lungen.deeplearning.net.autoencoder;

import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.graph.rnn.LastTimeStepVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.BatchNormalization;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.lungen.deeplearning.iterator.MultivariateIterator;
import org.lungen.deeplearning.listener.EarlyStopListener;
import org.lungen.deeplearning.listener.ScorePrintListener;
import org.lungen.deeplearning.listener.UIStatsListener;
import org.lungen.deeplearning.model.ModelPersistence;
import org.lungen.deeplearning.net.NeuralNet;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.HashMap;
import java.util.Map;

import static org.lungen.deeplearning.iterator.CharactersSets.*;

public class MultivariatePredictorNet implements NeuralNet {

    private static final Logger log = LoggerFactory.getLogger("net.multivariate");

    private ComputationGraph net;
    private EarlyStopListener earlyStopListener;
    private ScorePrintListener scorePrintListener;
    private MultivariateIterator iterator;
    private UIStatsListener statsListener;
    private String modelName;

    @Override
    public void init(Map<String, Object> params) {

        this.iterator = iterator(params);
        int numFeaturesRecurrent    = iterator.getNumSequenceFeatures();
        int numFeaturesNonRecurrent = iterator.getNumNonSequenceFeatures();

        this.modelName              = (String) params.get(PARAM_MODEL_NAME);
        double learningRate         = (Double) params.get(PARAM_LEARNING_RATE);
        double l2Regularization     = (Double) params.get(PARAM_L2_REGULARIZATION);
        int tbpttSize               = (Integer) params.get(PARAM_TRUNCATED_BPTT_SIZE);
        int numIterEarlyStop        = (Integer) params.get(PARAM_NUMBER_ITER_NO_IMPROVE_STOP);
        int minEpochsEarlyStop      = (Integer) params.getOrDefault(PARAM_MIN_EPOCHS_STOP, 0);

        final NeuralNetConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
//                .updater(new RmsProp(learningRate))
                .seed(7)
                .weightInit(WeightInit.XAVIER)
                .l2(l2Regularization)
                .updater(new Adam(learningRate))
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer);

        final int hiddenRecurrentSize = 250;
        final int hiddenNonRecurrentSize = 250;

        final ComputationGraphConfiguration.GraphBuilder graphBuilder = builder.graphBuilder()
                .backpropType(BackpropType.TruncatedBPTT)
                .tBPTTBackwardLength(tbpttSize)
                .tBPTTForwardLength(tbpttSize)
                .addInputs("recurrentInput", "nonRecurrentInput")
                .setInputTypes(
                        InputType.recurrent(numFeaturesRecurrent),
                        InputType.feedForward(numFeaturesNonRecurrent))
                .addLayer("encoder",
                        new LSTM.Builder()
                                .nIn(numFeaturesRecurrent)
                                .nOut(hiddenRecurrentSize)
                                .activation(Activation.TANH)
                                .build(),
                        "recurrentInput")
                .addVertex("thoughtVector",
                        new LastTimeStepVertex("recurrentInput"), "encoder")
                .addVertex("merge",
                        new MergeVertex(), "thoughtVector", "nonRecurrentInput")
                .addLayer("dense1", new DenseLayer.Builder()
                                .nIn(hiddenRecurrentSize + numFeaturesNonRecurrent)
                                .nOut(hiddenNonRecurrentSize)
                                .activation(Activation.RELU)
                                .build(),
                        "merge")
                .addLayer("batchNorm", new BatchNormalization.Builder()
                                .nIn(hiddenNonRecurrentSize)
                                .nOut(hiddenNonRecurrentSize)
                                .build(),
                        "dense1")
                .addLayer("dense2", new DenseLayer.Builder()
                                .nIn(hiddenNonRecurrentSize)
                                .nOut(1)
                                .activation(Activation.RELU)
                                .build(),
                        "batchNorm")
                .addLayer("output", new OutputLayer.Builder()
                                .activation(Activation.IDENTITY)
                                .lossFunction(LossFunctions.LossFunction.MSE)
                                .nOut(1)
                                .build(),
                        "dense2")
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
        int numEpochs                       = (Integer) params.get(PARAM_NUMBER_EPOCHS);
        int checkAfterNMinibatches          = (Integer) params.get(PARAM_CHECK_EACH_NUMBER_MINIBATCHES);
        int stopAfterNMinibatches           = (Integer) params.get(PARAM_STOP_AFTER_NUMBER_MINIBATCHES);

        // Do training, and then generate and print samples from network
        int miniBatchNumber = 0;
        boolean stop = false;
        for (int i = 0; i < numEpochs; i++) {
            scorePrintListener.setEpoch(i);
            earlyStopListener.setEpoch(i);

            log.info("[{}] Epoch started", i);

            while (iterator.hasNext()) {
                MultiDataSet ds = iterator.next();
                net.fit(ds);
                if (++miniBatchNumber % checkAfterNMinibatches == 0) {
                    log.info("--------------------\n");
                    log.info("Completed " + miniBatchNumber + " minibatches");

                    // sample
                    // sampler.output(ds);
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
            iterator.reset();
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
    public MultivariateIterator iterator(Map<String, Object> params) {
        String file         = (String) params.get(PARAM_DATA_FILE);
        int minibatchSize   = (Integer) params.get(PARAM_MINIBATCH_SIZE);

        return new MultivariateIterator(new File(file),
                minibatchSize, "description",
                createCharacterSet(LATIN, RUSSIAN, NUMBERS, PUNCTUATION, SPECIAL));
    }

    @Override
    public Map<String, Object> defaultParams() {
        Map<String, Object> params = new HashMap<>();
        params.put(PARAM_MODEL_NAME, "bugzilla");
        params.put(PARAM_DATA_FILE, "C:/DATA/Projects/DataSets/Bugzilla/bugzilla-processed-final.csv");
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
