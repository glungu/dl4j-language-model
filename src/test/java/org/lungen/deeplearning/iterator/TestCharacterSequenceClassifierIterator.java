package org.lungen.deeplearning.iterator;

import org.junit.Assert;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.io.File;
import java.util.Collections;
import java.util.List;

import static org.lungen.deeplearning.iterator.CharactersSets.RUSSIAN_LOWERCASE;
import static org.lungen.deeplearning.iterator.CharactersSets.createCharacterSet;

/**
 * TestCharacterSequenceClassifierIterator
 *
 * @author lungen.tech@gmail.com
 */
public class TestCharacterSequenceClassifierIterator {

    @Test
    public void testOnTestData() {
        String fileTest = "C:/DATA/Projects/DataSets/RU_Wiktionary/words.test.csv";
        CharacterSequenceClassifierIterator iter = new CharacterSequenceClassifierIterator(
                new File(fileTest),
                createCharacterSet(RUSSIAN_LOWERCASE, Collections.singletonList('-')),
                3, 1500);

        DataSet ds1 = iter.next();
        INDArray features = ds1.getFeatures();
        Assert.assertEquals(3, features.rank());
        Assert.assertEquals(1500, features.size(0));
        Assert.assertEquals(RUSSIAN_LOWERCASE.size() + 1, features.size(1));
        Assert.assertEquals(iter.getCharSequenceMaxLength(), features.size(2));

        // values
        Assert.assertEquals("горлодёр", getStringValue(iter, ds1, 0));
        Assert.assertEquals("торговка", getStringValue(iter, ds1, 1));
        Assert.assertEquals("взбаламутить", getStringValue(iter, ds1, 2));

        // compare lists
        List<DataSet> ds1List = ds1.asList();
        Assert.assertFalse(iter.hasNext());
        iter.reset();
        DataSet ds2 = iter.next();
        List<DataSet> ds2List = ds2.asList();
        Assert.assertEquals(ds1List, ds2List);
    }

    private String getStringValue(CharacterSequenceClassifierIterator iter, DataSet dataSet, int index) {
        INDArray features = dataSet.getFeatures();
        INDArray featuresMask = dataSet.getFeaturesMaskArray();

        INDArray array = features.get(NDArrayIndex.point(index), NDArrayIndex.all(), NDArrayIndex.all());
        INDArray maskArray = featuresMask.get(NDArrayIndex.point(index), NDArrayIndex.all());

        Assert.assertEquals(2, featuresMask.rank());
        Assert.assertEquals(1500, featuresMask.size(0));
        Assert.assertEquals(iter.getCharSequenceMaxLength(), featuresMask.size(1));

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

    @Test
    public void testOnTrainData() {
        String fileTrain = "C:/DATA/Projects/DataSets/RU_Wiktionary/words.train.csv";
        CharacterSequenceClassifierIterator iter = new CharacterSequenceClassifierIterator(
                new File(fileTrain),
                createCharacterSet(RUSSIAN_LOWERCASE, Collections.singletonList('-')),
                3, 64);


        int count = 0;
        while (iter.hasNext()) {
            DataSet ds = iter.next();
            INDArray features = ds.getFeatures();
            Assert.assertEquals(3, features.rank());
            Assert.assertEquals(iter.hasNext() ? 64 : 30000 % 64, features.size(0));
            Assert.assertEquals(RUSSIAN_LOWERCASE.size() + 1, features.size(1));
            Assert.assertEquals(iter.getCharSequenceMaxLength(), features.size(2));
            count++;
        }
        Assert.assertEquals((int) Math.ceil(30000 / (double) 64), count);
    }


}
