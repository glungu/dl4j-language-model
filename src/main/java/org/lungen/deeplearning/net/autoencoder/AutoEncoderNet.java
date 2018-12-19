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

    private ComputationGraph net;
    private EarlyStopListener earlyStopListener;
    private ScorePrintListener scorePrintListener;

    private AutoEncoderNet() {
    }

    public void init(Map<String, Object> initParams) {

        double learningRate     = (Double) initParams.get(PARAM_LEARNING_RATE);
        double l2Regularization = (Double) initParams.get(PARAM_L2_REGULARIZATION);
        int tbpttSize           = (Integer) initParams.get(PARAM_TRUNCATED_BPTT_SIZE);
        int numInputFeatures    = (Integer) initParams.get(PARAM_NUMBER_INPUT_FEATURES);

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

        net = new ComputationGraph(graphBuilder.build());
        earlyStopListener = new EarlyStopListener(100000);
        scorePrintListener = new ScorePrintListener(1);
        net.setListeners(scorePrintListener, earlyStopListener, new UIStatsListener());
        net.init();
    }

    @Override
    public void train(Map<String, Object> trainParams) {

        AutoEncoderCharacterIterator iterator    = (AutoEncoderCharacterIterator) trainParams.get(PARAM_ITERATOR);
        int numEpochs                       = (Integer) trainParams.get(PARAM_NUMBER_EPOCHS);
        int checkAfterNMinibatches          = (Integer) trainParams.get(PARAM_CHECK_EACH_NUMBER_MINIBATCHES);
        int stopAfterNMinibatches           = (Integer) trainParams.get(PARAM_STOP_AFTER_NUMBER_MINIBATCHES);

        AutoEncoderSampler sampler = new AutoEncoderSampler(net, iterator);

        // Do training, and then generate and print samples from network
        int miniBatchNumber = 0;
        boolean stop = false;
        for (int i = 0; i < numEpochs; i++) {
            scorePrintListener.setEpoch(i);
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
        log.info("Training complete!");

        ModelPersistence.save(net);
    }


    public static void main(String[] args) {
        File file = new File("C:/DATA/Projects/dl4j-language-model/src/main/resources/alice_in_wonderland.txt");
        AutoEncoderCharacterIterator iterator = new AutoEncoderCharacterIterator(file,
                32, 1000, CharactersSets.getEnglishCharacterSet());

        Map params = new HashMap();
        params.put(PARAM_ITERATOR, iterator);
        params.put(PARAM_LEARNING_RATE, 1e-3);
        params.put(PARAM_L2_REGULARIZATION, 1e-3);
        params.put(PARAM_TRUNCATED_BPTT_SIZE, 50);
        params.put(PARAM_CHECK_EACH_NUMBER_MINIBATCHES, 10);
        params.put(PARAM_STOP_AFTER_NUMBER_MINIBATCHES, -1);
        params.put(PARAM_NUMBER_INPUT_FEATURES, iterator.getDictionarySize());
        params.put(PARAM_NUMBER_EPOCHS, 2500);

        AutoEncoderNet net = new AutoEncoderNet();
        net.init(params);
        net.train(params);
    }
}
