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
import org.lungen.deeplearning.listener.EarlyStopListener;
import org.lungen.deeplearning.listener.ScorePrintListener;
import org.lungen.deeplearning.listener.UIStatsListener;
import org.lungen.deeplearning.net.NeuralNet;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Map;

public class MultivariatePredictorNet implements NeuralNet {

    private static final Logger log = LoggerFactory.getLogger("net.multivariate");

    private ComputationGraph net;
    private EarlyStopListener earlyStopListener;
    private ScorePrintListener scorePrintListener;

    @Override
    public void init(Map<String, Object> initParams) {

        double learningRate     = (Double) initParams.get(PARAM_LEARNING_RATE);
        double l2Regularization = (Double) initParams.get(PARAM_L2_REGULARIZATION);
        int tbpttSize           = (Integer) initParams.get(PARAM_TRUNCATED_BPTT_SIZE);
        int numFeaturesRecurrent    = (Integer) initParams.get(PARAM_NUMBER_INPUT_FEATURES);
        int numFeaturesNonRecurrent = (Integer) initParams.get(PARAM_NUMBER_INPUT_FEATURES);

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
                .backpropType(BackpropType.Standard)
                .tBPTTBackwardLength(tbpttSize)
                .tBPTTForwardLength(tbpttSize)
                .addInputs("recurrentInput", "nonRecurrentInput")
                .setInputTypes(
                        InputType.recurrent(numFeaturesRecurrent),
                        InputType.recurrent(numFeaturesNonRecurrent))
                .addLayer("encoder",
                        new LSTM.Builder()
                                .nIn(numFeaturesRecurrent)
                                .nOut(hiddenRecurrentSize)
                                .activation(Activation.TANH)
                                .build(),
                        "recurrentInput")
                .addVertex("thoughtVector",
                        new LastTimeStepVertex("encoderInput"), "encoder")
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
                                .build(),
                        "dense2")
                .setOutputs("output");

        net = new ComputationGraph(graphBuilder.build());
        earlyStopListener = new EarlyStopListener(100000);
        scorePrintListener = new ScorePrintListener(1);
        net.setListeners(scorePrintListener, earlyStopListener, new UIStatsListener());
        net.init();
    }

    @Override
    public void train(Map<String, Object> trainParams) {

    }
}
