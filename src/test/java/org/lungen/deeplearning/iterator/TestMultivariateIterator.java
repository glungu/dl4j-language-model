package org.lungen.deeplearning.iterator;

import org.junit.Assert;
import org.junit.Test;
import org.nd4j.linalg.dataset.api.MultiDataSet;

import java.io.File;

import static org.lungen.deeplearning.iterator.CharactersSets.*;
import static org.lungen.deeplearning.iterator.CharactersSets.PUNCTUATION;
import static org.lungen.deeplearning.iterator.CharactersSets.SPECIAL;

/**
 * TestMultivariateIterator
 *
 * @author lungen.tech@gmail.com
 */
public class TestMultivariateIterator {

    @Test
    public void test() {
        String dir = "C:/DATA/Projects/DataSets/Jnetx_Bugzilla";
        String filePath = dir + "/bugzilla-jnetx-processed-final.csv";
        MultivariateIterator iterator = new MultivariateIterator(new File(filePath),
                1, "description",
                createCharacterSet(LATIN, RUSSIAN, NUMBERS, PUNCTUATION, SPECIAL));
        MultiDataSet batch = iterator.next();
        Assert.assertEquals(2, batch.numFeatureArrays());
        Assert.assertEquals(1, batch.numLabelsArrays());

    }
}
