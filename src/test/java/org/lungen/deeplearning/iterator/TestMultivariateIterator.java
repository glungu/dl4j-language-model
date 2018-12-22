package org.lungen.deeplearning.iterator;

import org.junit.Assert;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.io.File;
import java.util.IntSummaryStatistics;
import java.util.stream.IntStream;

import static org.lungen.deeplearning.iterator.CharactersSets.*;

/**
 * TestMultivariateIterator
 *
 * @author lungen.tech@gmail.com
 */
public class TestMultivariateIterator {

    @Test
    public void test() {
        String dir = "C:/DATA/Projects/DataSets/Bugzilla";
        String filePath = dir + "/bugzilla-processed-final.csv";
        MultivariateIterator iterator = new MultivariateIterator(new File(filePath),
                1, "description",
                createCharacterSet(LATIN, RUSSIAN, NUMBERS, PUNCTUATION, SPECIAL));

        char[] seq = iterator.getSequence(iterator.getCurrentPosition());

        MultiDataSet batch = iterator.next();
        Assert.assertEquals(2, batch.numFeatureArrays());
        Assert.assertEquals(1, batch.numLabelsArrays());

        // mini batch size
        Assert.assertEquals(1, batch.getFeatures(0).size(0));

        INDArray sequence = batch.getFeatures(0).get(NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.all());
        Assert.assertEquals(2, sequence.rank());

        char[] strUnmasked = new char[(int) sequence.size(1)];
        for (int i = 0; i < sequence.size(1); i++) {
            INDArray charOneHot = sequence.get(NDArrayIndex.all(), NDArrayIndex.point(i));
            int argmax = (int) charOneHot.argMax().getDouble(0);
            strUnmasked[i] = iterator.indexToChar(argmax);
        }

        int[] mask = batch.getFeaturesMaskArray(0).get(NDArrayIndex.point(0), NDArrayIndex.all()).toIntVector();
        IntSummaryStatistics stats = IntStream.of(mask).summaryStatistics();
        Assert.assertEquals(1, stats.getMax());
        Assert.assertEquals(0, stats.getMin());

        StringBuilder strMasked = new StringBuilder();
        IntStream.range(0, strUnmasked.length)
                .filter(i -> mask[i] == 1)
                .forEach(i -> strMasked.append(strUnmasked[i]));
        Assert.assertEquals(String.valueOf(seq), strMasked.toString());

    }
}
