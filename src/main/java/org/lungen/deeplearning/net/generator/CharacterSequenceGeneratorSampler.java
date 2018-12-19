package org.lungen.deeplearning.net.generator;

import java.util.Arrays;
import java.util.Map;
import java.util.Random;
import java.util.stream.IntStream;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.lungen.deeplearning.iterator.CharacterIterator;
import org.lungen.deeplearning.model.ModelPersistence;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Created by user on 19.12.2017.
 */
public class CharacterSequenceGeneratorSampler {

    private static final double TINY = Double.MIN_VALUE;
    private static final double ONE = 1.0;

    private CharacterSequenceGeneratorSampler() {
    }

    /**
     * Generate a sample from the network, given an (optional, possibly null) initialization. Initialization
     * can be used to 'prime' the RNN with a sequence you want to extend/continue.<br>
     * Note that the initalization is used for all samples
     *
     * @param initialization     String, may be null. If null, select a random character as initialization for all samples
     * @param charactersToSample Number of characters to sample from network (excluding initialization)
     * @param net                MultiLayerNetwork with one or more GravesLSTM/RNN layers and a softmax output layer
     * @param iter               CharacterIterator. Used for going from indexes back to characters
     */
    public static String[] sample(
            String initialization, MultiLayerNetwork net,
            CharacterIterator iter, Random rng,
            int charactersToSample, int numSamples, double temperature) {

        if (temperature < 0.0 || temperature > 1.0) {
            throw new IllegalArgumentException("Temperature must be in range (0.0, 1.0]");
        }

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
        output = output.tensorAlongDimension((int) output.size(2) - 1, 1, 0);    //Gets the last time step output

        for (int i = 0; i < charactersToSample; i++) {
            //Set up next input (single time step) by sampling from previous output
            INDArray nextInput = Nd4j.zeros(numSamples, iter.inputColumns());
            //Output is a probability distribution. Sample from this for each example we want to generate, and add it to the new input
            for (int s = 0; s < numSamples; s++) {
                double[] outputProbDistribution = new double[iter.totalOutcomes()];
                for (int j = 0; j < outputProbDistribution.length; j++) {
                    outputProbDistribution[j] = output.getDouble(s, j);
                }
                int sampledCharacterIdx = temperature == ONE ?
                        sampleFromDistribution(outputProbDistribution, rng) :
                        sampleFromDistributionTemperature(outputProbDistribution, rng, temperature);

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

    private static String[] sample(
            String initialization, MultiLayerNetwork net,
            CharacterIterator iter, Random rng,
            int charactersToSample, int numSamples) {

        return sample(initialization, net, iter, rng, charactersToSample, numSamples, ONE);
    }

    private static int sampleFromDistributionTemperature(double[] distribution, Random rng, double temperature) {
        final double[] probs = new double[distribution.length];
        IntStream.range(0, distribution.length).forEach(i -> probs[i] = Math.exp(Math.log(distribution[i] + TINY) / temperature));
        double sum = Arrays.stream(probs).sum();
        IntStream.range(0, distribution.length).forEach(i -> probs[i] /= sum);
        return sampleFromDistribution(probs, rng);
    }

    /**
     * Given a probability distribution over discrete classes, sample from the distribution
     * and return the generated class index.
     *
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
                if (d <= sum) {
                    return i;
                }
            }
            // If we haven't found the right index yet, maybe the sum is slightly
            // lower than 1 due to rounding error, so try again.
        }
        // Should be extremely unlikely to happen if distribution is a valid probability distribution
        throw new IllegalArgumentException("Distribution is invalid? d=" + d + ", sum=" + sum);
    }

    public static void sampleToConsole(MultiLayerNetwork net,
                                       CharacterIterator iter,
                                       int miniBatchNumber,
                                       int nCharactersToSample,
                                       int nSamplesToGenerate,
                                       double temperature,
                                       Random rng) {

        int exampleLength = iter.getExampleLength();
        int miniBatchSize = iter.batch();

        StringBuilder str = new StringBuilder("--------------------");
        str.append("Completed ").append(miniBatchNumber).append(" minibatches of size ").append(miniBatchSize).append("x").append(exampleLength).append(" characters").append('\n');
        str.append("Sampling characters...").append('\n');
        String[] samples = sample(null, net, iter, rng, nCharactersToSample, nSamplesToGenerate, temperature);

        for (int j = 0; j < samples.length; j++) {
            str.append("----- Sample ").append(j).append(" -----").append('\n');
            str.append(samples[j]).append('\n').append('\n');
        }
        System.out.println(str.toString());
    }

    public static void main(String[] args) throws Exception {
        CharacterSequenceGeneratorNet tempNet = new CharacterSequenceGeneratorNet();
        Map<String, Object> params = tempNet.defaultParams();
        CharacterIterator iterator = tempNet.iterator(params);

        if (args == null || args.length == 0) {
            System.out.println("Persisted model must be provided as parameter");
            return;
        }

        String fileName = args[0];// "net-java-code-20181130-100817-score-34.47.zip";
        MultiLayerNetwork net = ModelPersistence.loadNet(fileName);

        String init = "package com.";
        System.out.print(init);
        System.out.flush();
        while (true) {
            String sample = sample(init, net, iterator, new Random(7),
                    1000, 1, 0.7)[0];
            for (char c : sample.substring(init.length()).toCharArray()) {
                System.out.print(c);
                System.out.flush();
                Thread.sleep(100);
            }
            init = sample;
        }

    }
}
