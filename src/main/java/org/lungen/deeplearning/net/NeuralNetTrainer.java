package org.lungen.deeplearning.net;

import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import java.util.stream.IntStream;

import org.lungen.deeplearning.net.generator.CharacterSequenceGeneratorNet;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static org.lungen.deeplearning.net.NeuralNet.*;

public class NeuralNetTrainer {

    private static final Logger log = LoggerFactory.getLogger("net.trainer");

    private NeuralNetTrainer() {
    }

    private static double trainOnce(double learningRate,
                                    double l2RegParam,
                                    int minEpochs) {

        log.info("### Starting... Learning rate: " + learningRate + ", L2 param: " + l2RegParam);

        NeuralNet net = new CharacterSequenceGeneratorNet();
        Map<String, Object> params = net.defaultParams();
        params.put(PARAM_MINIBATCH_SIZE, 64);
        params.put(PARAM_SEQUENCE_LENGTH, 5000);
        params.put(PARAM_LEARNING_RATE, learningRate);
        params.put(PARAM_L2_REGULARIZATION, l2RegParam);
        params.put(PARAM_TEMPERATURE, 0.9);
        params.put(PARAM_NUMBER_ITER_NO_IMPROVE_STOP, 10000);
        params.put(PARAM_MIN_EPOCHS_STOP, minEpochs);

        net.init(params);
        net.train(params);

        log.info("### Completed. Learning rate: " + learningRate + ", L2 param: " + l2RegParam + " -> " + net.getBestScore());

        return net.getBestScore();
    }

    public static void trainMultiple(int learningRateExpFrom, int learningRateExpTo,
                                     int l2RegularizationExpFrom, int l2RegularizationExpTo,
                                     int number) {
        Random rnd = new Random();
        double[] learningRates = getRandomExponentialValues(rnd, learningRateExpFrom, learningRateExpTo, number);
        double[] l2RegParams = getRandomExponentialValues(rnd, l2RegularizationExpFrom, l2RegularizationExpTo, number);

        Map<Tuple<Double, Double>, Double> resultScores = new HashMap<>();
        for (double learningRate : learningRates) {
            for (double l2RegParam : l2RegParams) {
                double score = trainOnce(learningRate, l2RegParam, 0);
                resultScores.put(new Tuple<>(learningRate, l2RegParam), score);
            }
        }

        log.info("### Final results of training with multiple hyper-parameters: ");
        resultScores.keySet().stream()
                .map(tuple -> "Learning rate: " + tuple.x + ", L2 param: " + tuple.y
                        + " -> " + resultScores.get(tuple))
                .forEach(log::info);
    }

    private static double[] getRandomExponentialValues(Random rnd, int expFrom, int expTo, int number) {
        int[] exponents = IntStream.range(expFrom, expTo).toArray();
        double[] values = new double[exponents.length * number];

        IntStream.range(0, exponents.length).forEachOrdered(i -> {
            IntStream.range(0, number).forEachOrdered(j -> {
                double coeff = j < number - 1 ? rnd.nextDouble() : 1.0;
                values[number * i + j] = Math.pow(10, exponents[i]) * coeff;
            });
        });

        return values;
    }

    private static final class Tuple<X, Y> {
        private final X x;
        private final Y y;

        private Tuple(X x, Y y) {
            this.x = x;
            this.y = y;
        }
    }

    public static void main(String[] args) {
//        NeuralNetTrainer.trainMultiple(new CharacterSequenceGeneratorFactoryJavaCode(),
//                -3, -2,
//                -3, -2);
        NeuralNetTrainer.trainOnce(
                1e-3,
                1e-4,
                1);
    }
}
