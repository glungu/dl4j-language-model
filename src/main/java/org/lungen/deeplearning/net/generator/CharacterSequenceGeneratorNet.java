package org.lungen.deeplearning.net.generator;

import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.lungen.deeplearning.iterator.CharacterIterator;
import org.lungen.deeplearning.iterator.CharacterIteratorFactory;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Random;

/**
 * RNN_WarAndPeace.
 * Recurrent network trained on Tolstoy's novel.
 *
 */
public class CharacterSequenceGeneratorNet {

    public static void main( String[] args ) throws Exception {
        int lstmLayerSize = 200;                    //Number of units in each GravesLSTM layer
        int miniBatchSize = 32;                        //Size of mini batch to use when  training
        int exampleLength = 1000;                    //Length of each training example sequence to use. This could certainly be increased
        int tbpttLength = 50;                       //Length for truncated backpropagation through time. i.e., do parameter updates ever 50 characters
        int numEpochs = 50;                            //Total number of training epochs
        int generateSamplesEveryNMinibatches = 10;  //How frequently to generate samples from the network? 1000 characters / 50 tbptt length: 20 parameter updates per minibatch
        int nSamplesToGenerate = 4;                    //Number of samples to generate after each training epoch
        int nCharactersToSample = 300;                //Length of each sample to generate
        String generationInitialization = null;        //Optional character initialization; a random character is used if null
        // Above is Used to 'prime' the LSTM with a character sequence to continue/complete.
        // Initialization characters must all be in CharacterIterator.getMinimalCharacterSet() by default
        Random rng = new Random(12345);

        //Get a DataSetIterator that handles vectorization of text into something we can use to train
        // our GravesLSTM network.
        CharacterIterator iter = CharacterIteratorFactory.getEnglishFileIterator(
                "alice_in_wonderland.txt", miniBatchSize, exampleLength);

        int nOut = iter.totalOutcomes();
        int nIn = iter.inputColumns();

        //Set up network configuration:
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(12345)
                .l2(0.001)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(0.005))
                .list()
                .layer(0, new LSTM.Builder().nIn(iter.inputColumns()).nOut(lstmLayerSize)
                        .activation(Activation.TANH).build())
//                .layer(1, new LSTM.Builder().nIn(lstmLayerSize).nOut(lstmLayerSize)
//                        .activation(Activation.TANH).build())
                .layer(1, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT).activation(Activation.SOFTMAX)        //MCXENT + softmax for classification
                        .nIn(lstmLayerSize).nOut(nOut).build())
                .backpropType(BackpropType.TruncatedBPTT).tBPTTForwardLength(tbpttLength).tBPTTBackwardLength(tbpttLength)
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        net.setListeners(new ScoreIterationListener(10), createStatsListenerForUI());

        //Print the  number of parameters in the network (and for each layer)
        Layer[] layers = net.getLayers();
        int totalNumParams = 0;
        for (int i = 0; i < layers.length; i++) {
            long nParams = layers[i].numParams();
            System.out.println("Number of parameters in layer " + i + ": " + nParams);
            totalNumParams += nParams;
        }
        System.out.println("Total number of network parameters: " + totalNumParams);

        //Do training, and then generate and print samples from network
        int miniBatchNumber = 0;
        for (int i = 0; i < numEpochs; i++) {
            while (iter.hasNext()) {
                DataSet ds = iter.next();
                net.fit(ds);
                if (++miniBatchNumber % generateSamplesEveryNMinibatches == 0) {
                    System.out.println("[" + i + "] --------------------");
                    System.out.println("Completed " + miniBatchNumber + " minibatches of size " + miniBatchSize + "x" + exampleLength + " characters");
                    System.out.println("Sampling characters from network given initialization \"" + (generationInitialization == null ? "" : generationInitialization) + "\"");
                    String[] samples = CharacterSequenceGeneratorSampler.sampleCharactersFromNetwork(
                            generationInitialization, net, iter, rng, nCharactersToSample, nSamplesToGenerate);

                    for (int j = 0; j < samples.length; j++) {
                        System.out.println("----- Sample " + j + " -----");
                        System.out.println(samples[j]);
                        System.out.println();
                    }
                }
            }

            iter.reset();    //Reset iterator for another epoch
            System.out.println("\n##### Epoch #" + i + " completed.\n");
        }

        System.out.println("\n\nExample complete");

        System.out.println("\n\nSaving Model");
        String currentDir = System.getProperty("user.dir");
        String date = new SimpleDateFormat("YYYYMMdd-HHmmss").format(new Date());
        String fileToSave = currentDir + "/src/main/resources/model-" + date + ".zip";

        //Where to save the network. Note: the file is in .zip format - can be opened externally
        ModelSerializer.writeModel(net, fileToSave, true);
        System.out.println("Model saved: " + fileToSave);

    }

    public static StatsListener createStatsListenerForUI() {
        //Initialize the user interface backend
        UIServer uiServer = UIServer.getInstance();

        //Configure where the network information (gradients, score vs. time etc)
        // is to be stored. Here: store in memory.
        //Alternative: new FileStatsStorage(File), for saving and loading later
        StatsStorage statsStorage = new InMemoryStatsStorage();

        //Attach the StatsStorage instance to the UI:
        // this allows the contents of the StatsStorage to be visualized
        uiServer.attach(statsStorage);

        //Then add the StatsListener to collect this information
        // from the network, as it trains
        return new StatsListener(statsStorage);
    }

}
