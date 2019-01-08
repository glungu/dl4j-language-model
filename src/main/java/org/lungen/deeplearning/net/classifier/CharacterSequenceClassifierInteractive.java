package org.lungen.deeplearning.net.classifier;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.lungen.deeplearning.iterator.CharacterSequenceClassifierIterator;
import org.lungen.deeplearning.model.ModelPersistence;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Map;
import java.util.stream.IntStream;

/**
 * CharacterSequenceClassifierInteractive
 *
 * @author lungen.tech@gmail.com
 */
public class CharacterSequenceClassifierInteractive {

    public static void main(String[] args) {
        CharacterSequenceClassifierNet netTemp = new CharacterSequenceClassifierNet();
        Map<String, Object> params = netTemp.defaultParams();
        CharacterSequenceClassifierIterator iterator = netTemp.iterator(params);
        CharacterSequenceClassifierIterator iteratorTest = netTemp.iteratorTest;

        ComputationGraph net = ModelPersistence.loadGraph("net-wiktionary-20191231-144757-score-0.78.zip");
//        Evaluation evaluation = net.evaluate(iteratorTest);
//        System.out.println("Accuracy: " + evaluation.accuracy());


        String[] words = {
                "кракозябра",
                "недодуривать",
                "фейсбучный",
                "колбасить",
                "сглючить",
                "бармаглот",
                "кнопаристый",
                "колобродский"
        };
        INDArray[] arrays = iterator.wordsToInputArray(words);
        INDArray[] output = net.output(false, new INDArray[]{arrays[0]}, new INDArray[]{arrays[1]});
        Integer[] result = iterator.outputArrayToLabels(output[0]);

        IntStream.range(0, words.length).forEach(i -> {
            System.out.println(words[i] + ": " + result[i]);
        });
    }
}
