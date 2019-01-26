package org.lungen.deeplearning.iterator;

import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.IntStream;

import static org.lungen.deeplearning.iterator.CharactersSets.*;

/**
 * TestCharacterSequenceValuePredictorIterator
 *
 * @author lungen.tech@gmail.com
 */
public class TestStringClassifierIterator {

    private StringClassifierIterator iterator;
    private int charSequenceMaxLength;
    private int miniBatchSize;

    @Before
    public void init() {
        String file = "C:/DATA/Projects/DataSets/Bugzilla/bugzilla.train.new.csv";
        charSequenceMaxLength = 2000;
        miniBatchSize = 32;

        iterator = new StringClassifierIterator(new File(file),
                createCharacterSet(RUSSIAN, LATIN, NUMBERS, PUNCTUATION, SPECIAL),
                charSequenceMaxLength, miniBatchSize);
    }

    @Test
    public void testIterator() {

        IntStream.range(0, iterator.getTotalSequences())
                .forEach(i -> Assert.assertTrue(iterator.getSequence(i).length <= charSequenceMaxLength));

        int count = 0;
        while (iterator.hasNext()) {
            DataSet ds = iterator.next();
            // print first batch
            // boolean printCondition = iterator.getCurrentPosition() == miniBatchSize;
            boolean printCondition = true;
            if (printCondition) {
                List<String> list = getStringValues(iterator, ds);
                list.forEach(System.out::println);
            }
            count++;
        }
        Assert.assertEquals((int) Math.ceil(iterator.getTotalSequences() / (double) miniBatchSize), count);
    }

    private List<String> getStringValues(StringClassifierIterator iter, DataSet dataSet) {
        List<String> result = new ArrayList<>();
        INDArray features = dataSet.getFeatures();
        INDArray featuresMask = dataSet.getFeaturesMaskArray();
        INDArray labels = dataSet.getLabels();

        Assert.assertEquals(3, features.rank());
        Assert.assertEquals(2, featuresMask.rank());

        int size = iterator.hasNext() ? miniBatchSize : iterator.getTotalSequences() - iterator.getCurrentPosition();
        Assert.assertEquals(size, features.size(0));
        Assert.assertEquals(size, featuresMask.size(0));
        Assert.assertEquals(size, labels.size(0));

        Assert.assertEquals(iter.getNumSequenceFeatures(), features.size(1));
        Assert.assertEquals(iter.getCharSequenceMaxLength(), features.size(2));
        Assert.assertEquals(iter.getCharSequenceMaxLength(), featuresMask.size(1));

        // iterate over examples
        for (int index = 0; index < features.size(0); index++) {
            result.add(getStringValue(iter, features, featuresMask, index));
        }
        return result;
    }

    private String getStringValue(StringClassifierIterator iter, INDArray features, INDArray featuresMask, int index) {
        INDArray array = features.get(NDArrayIndex.point(index), NDArrayIndex.all(), NDArrayIndex.all());
        INDArray maskArray = featuresMask.get(NDArrayIndex.point(index), NDArrayIndex.all());

        StringBuilder str = new StringBuilder();
        for (int i = 0; i < array.size(1); i++) {
            if (maskArray.getInt(i) == 1) {
                INDArray oneHotChar = array.get(NDArrayIndex.all(), NDArrayIndex.point(i));
                char c = iter.indexToChar(oneHotChar.argMax(0).getInt(0));
                str.append(c);
            }
        }
        return str.toString();
    }

}
