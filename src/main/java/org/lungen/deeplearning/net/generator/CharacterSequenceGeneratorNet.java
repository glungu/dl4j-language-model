package org.lungen.deeplearning.net.generator;

import java.io.File;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.lungen.deeplearning.iterator.CharacterIterator;
import org.lungen.deeplearning.iterator.CharactersSets;
import org.lungen.deeplearning.listener.EarlyStopListener;
import org.lungen.deeplearning.listener.ScorePrintListener;
import org.lungen.deeplearning.listener.UIStatsListener;
import org.lungen.deeplearning.model.ModelPersistence;
import org.lungen.deeplearning.net.NeuralNet;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Character sequence generation network.
 *
 */
public class CharacterSequenceGeneratorNet implements NeuralNet {

    private static final Logger log = LoggerFactory.getLogger("net.generator");

    private ScorePrintListener scorePrintListener;
    private MultiLayerNetwork net;
    private EarlyStopListener earlyStopListener;

    @Override
    public void init(Map<String, Object> initParams) {

        double learningRate     = (Double) initParams.get(PARAM_LEARNING_RATE);
        double l2Regularization = (Double) initParams.get(PARAM_L2_REGULARIZATION);
        int tbpttSize           = (Integer) initParams.get(PARAM_TRUNCATED_BPTT_SIZE);
        int numInputFeatures    = (Integer) initParams.get(PARAM_NUMBER_INPUT_FEATURES);

        int lstmLayerSize = 200;

        //Set up network configuration:
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(12345)
                .l2(l2Regularization)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(learningRate))
                .list()
                .layer(0, new LSTM.Builder().nIn(numInputFeatures).nOut(lstmLayerSize)
                        .activation(Activation.TANH).build())
//                .layer(1, new LSTM.Builder().nIn(lstmLayerSize).nOut(lstmLayerSize)
//                        .activation(Activation.TANH).build())
                .layer(1, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT).activation(Activation.SOFTMAX)        //MCXENT + softmax for classification
                        .nIn(lstmLayerSize).nOut(numInputFeatures).build())
                .backpropType(BackpropType.TruncatedBPTT).tBPTTForwardLength(tbpttSize).tBPTTBackwardLength(tbpttSize)
                .build();

        net = new MultiLayerNetwork(conf);
        net.init();
        scorePrintListener = new ScorePrintListener(10);
        UIStatsListener uiStatsListener = new UIStatsListener();
        earlyStopListener = new EarlyStopListener(10000);
        net.setListeners(scorePrintListener, uiStatsListener, earlyStopListener);
    }

    @Override
    public void train(Map<String, Object> trainParams) {

        CharacterIterator iter      = (CharacterIterator) trainParams.get(PARAM_ITERATOR);
        int numEpochs               = (Integer) trainParams.get(PARAM_NUMBER_EPOCHS);
        int checkAfterNMinibatches  = (Integer) trainParams.get(PARAM_CHECK_EACH_NUMBER_MINIBATCHES);

        // Do training, and then generate and print samples from network
        Random rng = new Random(7);
        int nCharactersToSample = 1000;
        int nSamplesToGenerate = 1;

        int miniBatchNumber = 0;
        boolean stop = false;
        for (int i = 0; i < numEpochs; i++) {
            scorePrintListener.setEpoch(i);
            while (iter.hasNext()) {
                DataSet ds = iter.next();
                net.fit(ds);
                if (++miniBatchNumber % checkAfterNMinibatches == 0) {
                    CharacterSequenceGeneratorSampler.sampleToConsole(net, iter,
                            miniBatchNumber, nCharactersToSample, nSamplesToGenerate, rng);
                }
                if (earlyStopListener.isStopRecommended()) {
                    stop = true;
                    break;
                }
            }
            if (stop) {
                break;
            }

            // Reset iterator for another epoch
            iter.reset();
        }
        log.info("Training complete");

        ModelPersistence.save(net);
    }

    public static void main(String[] args ) throws Exception {
        File file = new File("C:/Work/ML/code/jnetx-code-java.txt");
        CharacterIterator iterator = new CharacterIterator(file,
                32, 1000, CharactersSets.getEnglishCharacterSet());

        Map params = new HashMap();
        params.put(PARAM_ITERATOR, iterator);
        params.put(PARAM_LEARNING_RATE, 1e-3);
        params.put(PARAM_L2_REGULARIZATION, 1e-3);
        params.put(PARAM_TRUNCATED_BPTT_SIZE, 50);
        params.put(PARAM_CHECK_EACH_NUMBER_MINIBATCHES, 10);
        params.put(PARAM_NUMBER_INPUT_FEATURES, iterator.inputColumns());
        params.put(PARAM_NUMBER_EPOCHS, 3);

        CharacterSequenceGeneratorNet net = new CharacterSequenceGeneratorNet();
        net.init(params);
        net.train(params);
    }
}
