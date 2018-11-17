package org.lungen.deeplearning.net.generator;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.lungen.deeplearning.iterator.CharacterIterator;
import org.lungen.deeplearning.iterator.CharacterIteratorFactory;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Random;

/**
 * Created by user on 19.12.2017.
 */
public class CharacterSequenceGeneratorSampler {

    public static String sample(
            String initialization, MultiLayerNetwork net,
            CharacterIterator iter, Random rng,
            int charactersToSample) {

        // int numSamples = 1;

        //Set up initialization. If no initialization: use a random character
        if (initialization == null) {
            initialization = String.valueOf(iter.getRandomCharacter());
        }

        //Create input for initialization
        INDArray initializationInput = Nd4j.zeros(1, iter.inputColumns(), initialization.length());
        char[] init = initialization.toCharArray();
        for (int i = 0; i < init.length; i++) {
            int idx = iter.convertCharacterToIndex(init[i]);
            // for (int j = 0; j < numSamples; j++) {
                initializationInput.putScalar(new int[]{0, idx, i}, 1.0f);
            // }
        }

        StringBuilder sb = new StringBuilder(initialization);

        //Sample from network (and feed samples back into input) one character at a time (for all samples)
        //Sampling is done in parallel here
        net.rnnClearPreviousState();
        INDArray output = net.rnnTimeStep(initializationInput);
        output = output.tensorAlongDimension((int)output.size(2) - 1, 1, 0);    //Gets the last time step output

        for (int i = 0; i < charactersToSample; i++) {
            //Set up next input (single time step) by sampling from previous output
            INDArray nextInput = Nd4j.zeros(1, iter.inputColumns());
            //Output is a probability distribution. Sample from this for each example we want to generate, and add it to the new input
            //for (int s = 0; s < numSamples; s++) {
                double[] outputProbDistribution = new double[iter.totalOutcomes()];
                for (int j = 0; j < outputProbDistribution.length; j++) {
                    outputProbDistribution[j] = output.getDouble(0, j);
                }
                int sampledCharacterIdx = sampleFromDistribution(outputProbDistribution, rng);

                nextInput.putScalar(new int[]{0, sampledCharacterIdx}, 1.0f);        //Prepare next time step input
                sb.append(iter.convertIndexToCharacter(sampledCharacterIdx));    //Add sampled character to StringBuilder (human readable output)
            //}

            output = net.rnnTimeStep(nextInput);    //Do one time step of forward pass
        }

        return sb.toString();
    }


    /** Generate a sample from the network, given an (optional, possibly null) initialization. Initialization
     * can be used to 'prime' the RNN with a sequence you want to extend/continue.<br>
     * Note that the initalization is used for all samples
     * @param initialization String, may be null. If null, select a random character as initialization for all samples
     * @param charactersToSample Number of characters to sample from network (excluding initialization)
     * @param net MultiLayerNetwork with one or more GravesLSTM/RNN layers and a softmax output layer
     * @param iter CharacterIterator. Used for going from indexes back to characters
     */
    public static String[] sampleCharactersFromNetwork(
            String initialization, MultiLayerNetwork net,
            CharacterIterator iter, Random rng,
            int charactersToSample, int numSamples) {

        //Set up initialization. If no initialization: use a random character
        if (initialization == null) {
            initialization = String.valueOf(iter.getRandomCharacter());
        }

        //Create input for initialization
        INDArray initializationInput = Nd4j.zeros(numSamples, iter.inputColumns(), initialization.length());
        char[] init = initialization.toCharArray();
        for (int i = 0; i < init.length; i++) {
            int idx = iter.convertCharacterToIndex(init[i]);
            for (int j = 0; j < numSamples; j++) {
                initializationInput.putScalar(new int[]{j, idx, i}, 1.0f);
            }
        }

        StringBuilder[] sb = new StringBuilder[numSamples];
        for (int i = 0; i < numSamples; i++) {
            sb[i] = new StringBuilder(initialization);
        }

        //Sample from network (and feed samples back into input) one character at a time (for all samples)
        //Sampling is done in parallel here
        net.rnnClearPreviousState();
        INDArray output = net.rnnTimeStep(initializationInput);
        output = output.tensorAlongDimension((int)output.size(2) - 1, 1, 0);    //Gets the last time step output

        for (int i = 0; i < charactersToSample; i++) {
            //Set up next input (single time step) by sampling from previous output
            INDArray nextInput = Nd4j.zeros(numSamples, iter.inputColumns());
            //Output is a probability distribution. Sample from this for each example we want to generate, and add it to the new input
            for (int s = 0; s < numSamples; s++) {
                double[] outputProbDistribution = new double[iter.totalOutcomes()];
                for (int j = 0; j < outputProbDistribution.length; j++) {
                    outputProbDistribution[j] = output.getDouble(s, j);
                }
                int sampledCharacterIdx = sampleFromDistribution(outputProbDistribution, rng);

                nextInput.putScalar(new int[]{s, sampledCharacterIdx}, 1.0f);        //Prepare next time step input
                sb[s].append(iter.convertIndexToCharacter(sampledCharacterIdx));    //Add sampled character to StringBuilder (human readable output)
            }

            output = net.rnnTimeStep(nextInput);    //Do one time step of forward pass
        }

        String[] out = new String[numSamples];
        for (int i = 0; i < numSamples; i++) {
            out[i] = sb[i].toString();
        }
        return out;
    }

    /** Given a probability distribution over discrete classes, sample from the distribution
     * and return the generated class index.
     * @param distribution Probability distribution over classes. Must sum to 1.0
     */
    private static int sampleFromDistribution(double[] distribution, Random rng) {
        double d = 0.0;
        double sum = 0.0;
        for (int t = 0; t < 10; t++) {
            d = rng.nextDouble();
            sum = 0.0;
            for (int i = 0; i < distribution.length; i++) {
                sum += distribution[i];
                if (d <= sum) return i;
            }
            //If we haven't found the right index yet, maybe the sum is slightly
            //lower than 1 due to rounding error, so try again.
        }
        //Should be extremely unlikely to happen if distribution is a valid probability distribution
        throw new IllegalArgumentException("Distribution is invalid? d=" + d + ", sum=" + sum);
    }

    public static void main(String[] args) throws Exception {
        // sample from saved model
        Random rng = new Random(12345);
        int miniBatchSize = 32;
        int exampleLength = 1000;
        int nCharactersToSample = 1000;

        System.out.println("\n\nLoading Model");
        String currentDir = System.getProperty("user.dir");
        String fileLocation = currentDir + "/src/main/resources/model-20181108-195958.zip";
        System.out.println("### " + fileLocation);

        //Load the model
        MultiLayerNetwork net = ModelSerializer.restoreMultiLayerNetwork(fileLocation);

        CharacterIterator iter = CharacterIteratorFactory.getEnglishFileIterator(
                "alice_in_wonderland.txt", miniBatchSize, exampleLength);

        String sample = CharacterSequenceGeneratorSampler.sample(
                "Alice said, ", net, iter, rng, nCharactersToSample);

        System.out.println(sample);
        System.out.println();
    }


}
