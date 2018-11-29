package org.lungen.deeplearning.net.generator;

import java.io.File;
import java.io.IOException;
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

    private String modelName;
    private MultiLayerNetwork net;
    private ScorePrintListener scorePrintListener;
    private EarlyStopListener earlyStopListener;
    private UIStatsListener statsListener;
    private CharacterIterator iterator;

    @Override
    public void init(Map<String, Object> params) {

        iterator = iterator(params);

        String modelName        = (String) params.get(PARAM_MODEL_NAME);
        double learningRate     = (Double) params.get(PARAM_LEARNING_RATE);
        double l2Regularization = (Double) params.get(PARAM_L2_REGULARIZATION);
        int tbpttSize           = (Integer) params.get(PARAM_TRUNCATED_BPTT_SIZE);
        int numInputFeatures    = iterator.inputColumns();
        int numIterEarlyStop    = (Integer) params.get(PARAM_NUMBER_ITER_NO_IMPROVE_STOP);
        int minEpochsEarlyStop  = (Integer) params.getOrDefault(PARAM_MIN_EPOCHS_STOP, 0);

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
                .layer(1, new LSTM.Builder().nIn(lstmLayerSize).nOut(lstmLayerSize)
                        .activation(Activation.TANH).build())
                .layer(2, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT).activation(Activation.SOFTMAX)        //MCXENT + softmax for classification
                        .nIn(lstmLayerSize).nOut(numInputFeatures).build())
                .backpropType(BackpropType.TruncatedBPTT).tBPTTForwardLength(tbpttSize).tBPTTBackwardLength(tbpttSize)
                .build();

        this.modelName = modelName;
        this.net = new MultiLayerNetwork(conf);
        this.net.init();
        this.scorePrintListener = new ScorePrintListener(10);
        this.statsListener = new UIStatsListener();
        this.earlyStopListener = new EarlyStopListener(modelName, numIterEarlyStop, minEpochsEarlyStop);
        this.net.setListeners(scorePrintListener, statsListener, earlyStopListener);
    }

    @Override
    public void train(Map<String, Object> params) {

        int numEpochs               = (Integer) params.get(PARAM_NUMBER_EPOCHS);
        int checkAfterNMinibatches  = (Integer) params.get(PARAM_CHECK_EACH_NUMBER_MINIBATCHES);
        double temperature          = (Double) params.get(PARAM_TEMPERATURE);

        // Do training, and then generate and print samples from network
        Random rng = new Random(7);
        int nCharactersToSample = 1000;
        int nSamplesToGenerate = 1;

        int miniBatchNumber = 0;
        boolean stop = false;
        for (int i = 0; i < numEpochs; i++) {
            scorePrintListener.setEpoch(i);
            earlyStopListener.setEpoch(i);

            while (iterator.hasNext()) {
                DataSet ds = iterator.next();
                net.fit(ds);
                if (++miniBatchNumber % checkAfterNMinibatches == 0) {
                    CharacterSequenceGeneratorSampler.sampleToConsole(net, iterator,
                            miniBatchNumber, nCharactersToSample, nSamplesToGenerate, temperature, rng);
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
            iterator.reset();
        }
        statsListener.close();
        log.info("Training complete");

        ModelPersistence.save(modelName, net);
    }

    @Override
    public CharacterIterator iterator(Map<String, Object> params) {
        String file         = (String) params.get(PARAM_DATA_FILE);
        int minibatchSize   = (Integer) params.get(PARAM_MINIBATCH_SIZE);
        int sequenceLength  = (Integer) params.get(PARAM_SEQUENCE_LENGTH);

        try {
            char[] chars = CharactersSets.getEnglishExtendedCharacterSet();
            return new CharacterIterator(new File(file), minibatchSize, sequenceLength, chars);
        } catch (IOException e) {
            throw new IllegalArgumentException("Cannot create iterator", e);
        }
    }

    @Override
    public double getBestScore() {
        return earlyStopListener.getBestScore();
    }

    @Override
    public Map<String, Object> defaultParams() {
        Map<String, Object> params = new HashMap<>();
        params.put(PARAM_MODEL_NAME, "jnetx-code");
        params.put(PARAM_DATA_FILE, "C:/Work/ML/code/jnetx-code-java.txt");
        params.put(PARAM_MINIBATCH_SIZE, 32);
        params.put(PARAM_SEQUENCE_LENGTH, 2000);
        params.put(PARAM_LEARNING_RATE, 5e-3);
        params.put(PARAM_L2_REGULARIZATION, 1e-3);
        params.put(PARAM_TRUNCATED_BPTT_SIZE, 50);
        params.put(PARAM_CHECK_EACH_NUMBER_MINIBATCHES, 10);
        params.put(PARAM_NUMBER_EPOCHS, 3);
        params.put(PARAM_NUMBER_ITER_NO_IMPROVE_STOP, 20000);
        return params;
    }

    public static void main(String[] args ) {
        CharacterSequenceGeneratorNet net = new CharacterSequenceGeneratorNet();
        Map<String, Object> params = net.defaultParams();
        net.init(params);
        net.train(params);
    }
}
