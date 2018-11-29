package org.lungen.deeplearning.net.autoencoder;

import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.graph.rnn.DuplicateToTimeSeriesVertex;
import org.deeplearning4j.nn.conf.graph.rnn.LastTimeStepVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.lungen.deeplearning.iterator.AutoEncoderCharacterIterator;
import org.lungen.deeplearning.iterator.CharactersSets;
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


public class AutoEncoderNet implements NeuralNet {

    private static final Logger log = LoggerFactory.getLogger("autoencoder.net");

    private String modelName;
    private ComputationGraph net;
    private EarlyStopListener earlyStopListener;
    private ScorePrintListener scorePrintListener;
    private UIStatsListener statsListener;
    private AutoEncoderCharacterIterator iterator;

    private AutoEncoderNet() {
    }

    public void init(Map<String, Object> params) {

        iterator = iterator(params);

        String modelName        = (String) params.get(PARAM_MODEL_NAME);
        double learningRate     = (Double) params.get(PARAM_LEARNING_RATE);
        double l2Regularization = (Double) params.get(PARAM_L2_REGULARIZATION);
        int tbpttSize           = (Integer) params.get(PARAM_TRUNCATED_BPTT_SIZE);
        int numInputFeatures    = iterator.getDictionarySize();
        int numIterEarlyStop    = (Integer) params.get(PARAM_NUMBER_ITER_NO_IMPROVE_STOP);
        int minEpochsEarlyStop  = (Integer) params.getOrDefault(PARAM_MIN_EPOCHS_STOP, 0);


        final NeuralNetConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
//                .updater(new RmsProp(learningRate))
                .seed(7)
                .weightInit(WeightInit.XAVIER)
                .l2(l2Regularization)
                .updater(new Adam(learningRate))
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer);

        final int hiddenLayerSize = 250;

        final ComputationGraphConfiguration.GraphBuilder graphBuilder = builder.graphBuilder()
                .backpropType(BackpropType.Standard)
                .tBPTTBackwardLength(tbpttSize)
                .tBPTTForwardLength(tbpttSize)
                .addInputs("encoderInput", "decoderInput")
                .setInputTypes(InputType.recurrent(numInputFeatures), InputType.recurrent(numInputFeatures))
                .addLayer("encoder",
                        new LSTM.Builder()
                                .nIn(numInputFeatures)
                                .nOut(hiddenLayerSize)
                                .activation(Activation.TANH)
                                .build(),
                        "encoderInput")
                .addVertex("thoughtVector",
                        new LastTimeStepVertex("encoderInput"), "encoder")
                .addVertex("duplication",
                        new DuplicateToTimeSeriesVertex("decoderInput"), "thoughtVector")
                .addVertex("merge",
                        new MergeVertex(), "decoderInput", "duplication")
                .addLayer("decoder",
                        new LSTM.Builder()
                                .nIn(numInputFeatures + hiddenLayerSize)
                                .nOut(hiddenLayerSize)
                                .activation(Activation.TANH)
                                .build(),
                        "merge")
                .addLayer("output",
                        new RnnOutputLayer.Builder()
                                .nIn(hiddenLayerSize)
                                .nOut(numInputFeatures)
                                .activation(Activation.SOFTMAX)
                                .lossFunction(LossFunctions.LossFunction.MCXENT)
                                .build(),
                        "decoder")
                .setOutputs("output");

        this.modelName = modelName;
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

        AutoEncoderSampler sampler = new AutoEncoderSampler(net, iterator);

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
                    log.info("Completed " + miniBatchNumber + " minibatches of size " + iterator.getMiniBatchSize() + "x" + iterator.getExampleLength() + " characters\n");

                    // sample
                    sampler.output(ds);
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
        return earlyStopListener.getBestScore();
    }

    @Override
    public AutoEncoderCharacterIterator iterator(Map<String, Object> params) {
        String file         = (String) params.get(PARAM_DATA_FILE);
        int minibatchSize   = (Integer) params.get(PARAM_MINIBATCH_SIZE);
        int sequenceLength  = (Integer) params.get(PARAM_SEQUENCE_LENGTH);

        return new AutoEncoderCharacterIterator(new File(file),
                minibatchSize,
                sequenceLength,
                CharactersSets.getEnglishCharacterSet());

    }

    @Override
    public Map<String, Object> defaultParams() {
        Map<String, Object> params = new HashMap<>();
        params.put(PARAM_MODEL_NAME, "alice-autoencoder");
        params.put(PARAM_DATA_FILE, "C:/DATA/Projects/dl4j-language-model/src/main/resources/alice_in_wonderland.txt");
        params.put(PARAM_MINIBATCH_SIZE, 32);
        params.put(PARAM_SEQUENCE_LENGTH, 1000);
        params.put(PARAM_LEARNING_RATE, 1e-3);
        params.put(PARAM_L2_REGULARIZATION, 1e-3);
        params.put(PARAM_TRUNCATED_BPTT_SIZE, 50);
        params.put(PARAM_CHECK_EACH_NUMBER_MINIBATCHES, 10);
        params.put(PARAM_STOP_AFTER_NUMBER_MINIBATCHES, -1);
        params.put(PARAM_NUMBER_EPOCHS, 200);
        return params;
    }

    public static void main(String[] args) {
        AutoEncoderNet net = new AutoEncoderNet();
        Map<String, Object> params = net.defaultParams();
        net.init(params);
        net.train(params);
    }
}
