package org.lungen.deeplearning.net.classifier;

import java.io.File;
import java.util.HashMap;
import java.util.Map;

import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader;
import org.datavec.api.split.NumberedFileInputSplit;
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.lungen.deeplearning.listener.EarlyStopListener;
import org.lungen.deeplearning.listener.ScorePrintListener;
import org.lungen.deeplearning.listener.UIStatsListener;
import org.lungen.deeplearning.model.ModelPersistence;
import org.lungen.deeplearning.net.NeuralNet;
import org.lungen.deeplearning.net.generator.CharacterSequenceGeneratorNet;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.primitives.Pair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class SequenceClassifierNet implements NeuralNet {

    private static File baseDir = new File("C:/Work/ML/uci/");
    private static File baseTrainDir = new File(baseDir, "train");
    private static File featuresDirTrain = new File(baseTrainDir, "features");
    private static File labelsDirTrain = new File(baseTrainDir, "labels");
    private static File baseTestDir = new File(baseDir, "test");
    private static File featuresDirTest = new File(baseTestDir, "features");
    private static File labelsDirTest = new File(baseTestDir, "labels");


    private static final Logger log = LoggerFactory.getLogger("net.classifier");

    private String modelName;
    private MultiLayerNetwork net;
    private ScorePrintListener scorePrintListener;
    private EarlyStopListener earlyStopListener;
    private UIStatsListener statsListener;
    private DataSetIterator iteratorTrain;
    private DataSetIterator iteratorTest;

    @Override
    public void init(Map<String, Object> params) {

        Pair<DataSetIterator, DataSetIterator> pair = iterator(params);
        this.iteratorTrain = pair.getFirst();
        this.iteratorTest = pair.getSecond();

        String modelName        = (String) params.get(PARAM_MODEL_NAME);
        double learningRate     = (Double) params.get(PARAM_LEARNING_RATE);
        double l2Regularization = (Double) params.get(PARAM_L2_REGULARIZATION);
        int tbpttSize           = (Integer) params.get(PARAM_TRUNCATED_BPTT_SIZE);
        int numIterEarlyStop    = (Integer) params.get(PARAM_NUMBER_ITER_NO_IMPROVE_STOP);
        int minEpochsEarlyStop  = (Integer) params.getOrDefault(PARAM_MIN_EPOCHS_STOP, 0);
        int numLabelClasses     = (Integer) params.get(PARAM_NUMBER_OUTPUT_CLASSES);

        int lstmLayerSize = 200;

        // ----- Configure the network -----
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(123)    //Random number generator seed for improved repeatability. Optional.
                .weightInit(WeightInit.XAVIER)
                .updater(new Nesterovs(0.005))
                .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)  //Not always required, but helps with this data set
                .gradientNormalizationThreshold(0.5)
                .list()
                .layer(0, new LSTM.Builder().activation(Activation.TANH).nIn(1).nOut(10).build())
                .layer(1, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX).nIn(10).nOut(numLabelClasses).build())
                .pretrain(false).backprop(true).build();

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

        // ----- Train the network, evaluating the test set performance at each epoch -----
        String str = "Test set evaluation at epoch %d: Accuracy = %.2f, F1 = %.2f";
        for (int i = 0; i < numEpochs; i++) {
            scorePrintListener.setEpoch(i);
            earlyStopListener.setEpoch(i);

            // actual training
            net.fit(iteratorTrain);

            //Evaluate on the test set:
            Evaluation evaluation = net.evaluate(iteratorTest);
            log.info(String.format(str, i, evaluation.accuracy(), evaluation.f1()));

            if (earlyStopListener.isStopRecommended()) {
                break;
            }

            iteratorTrain.reset();
            iteratorTest.reset();
        }

        statsListener.close();
        log.info("Training complete");

        ModelPersistence.save(modelName, net);
    }

    @Override
    public Pair<DataSetIterator, DataSetIterator> iterator(Map<String, Object> params) {
        String file         = (String) params.get(PARAM_DATA_FILE);
        int miniBatchSize   = (Integer) params.get(PARAM_MINIBATCH_SIZE);
        int sequenceLength  = (Integer) params.get(PARAM_SEQUENCE_LENGTH);

        try {
            // ----- Load the training data -----
            //Note that we have 450 training files for features: train/features/0.csv through train/features/449.csv
            SequenceRecordReader trainFeatures = new CSVSequenceRecordReader();
            trainFeatures.initialize(new NumberedFileInputSplit(
                    featuresDirTrain.getAbsolutePath() + "/%d.csv", 0, 449));
            SequenceRecordReader trainLabels = new CSVSequenceRecordReader();
            trainLabels.initialize(new NumberedFileInputSplit(
                    labelsDirTrain.getAbsolutePath() + "/%d.csv", 0, 449));

            int numLabelClasses = (Integer) params.get(PARAM_NUMBER_OUTPUT_CLASSES);

            DataSetIterator trainData = new SequenceRecordReaderDataSetIterator(
                    trainFeatures, trainLabels, miniBatchSize, numLabelClasses, false,
                    SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);

            //Normalize the training data
            DataNormalization normalizer = new NormalizerStandardize();
            normalizer.fit(trainData);              //Collect training data statistics
            trainData.reset();

            //Use previously collected statistics to normalize on-the-fly. Each DataSet returned by 'trainData' iterator will be normalized
            trainData.setPreProcessor(normalizer);


            // ----- Load the test data -----
            //Same process as for the training data.
            SequenceRecordReader testFeatures = new CSVSequenceRecordReader();
            testFeatures.initialize(new NumberedFileInputSplit(featuresDirTest.getAbsolutePath() + "/%d.csv", 0, 149));
            SequenceRecordReader testLabels = new CSVSequenceRecordReader();
            testLabels.initialize(new NumberedFileInputSplit(labelsDirTest.getAbsolutePath() + "/%d.csv", 0, 149));

            DataSetIterator testData = new SequenceRecordReaderDataSetIterator(testFeatures, testLabels, miniBatchSize, numLabelClasses,
                    false, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);

            testData.setPreProcessor(normalizer);   //Note that we are using the exact same normalization process as the training data
            return new Pair<DataSetIterator, DataSetIterator>(trainData, testData);

        } catch (Exception e) {
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
        params.put(PARAM_MODEL_NAME, "uci");
        params.put(PARAM_MINIBATCH_SIZE, 10);
        params.put(PARAM_NUMBER_OUTPUT_CLASSES, 6);
        params.put(PARAM_CHECK_EACH_NUMBER_MINIBATCHES, 10);
        params.put(PARAM_NUMBER_EPOCHS, 40);
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

