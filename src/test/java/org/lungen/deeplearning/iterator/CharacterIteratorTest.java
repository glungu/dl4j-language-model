package org.lungen.deeplearning.iterator;

import org.junit.Assert;
import org.junit.Test;

import java.io.File;
import java.net.URL;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.util.Random;

/**
 * Unit-test for CharacterIteratorFactory
 */
public class CharacterIteratorTest {

    @Test
    public void testSize() throws Exception {
        int miniBatchSize = 16;
        int exampleLength = 500;
        File f = new File(CharacterIterator.class.getResource("/tolstoy_selected.txt").toURI());
        CharacterIterator iter = new CharacterIterator(f.getAbsolutePath(),
                Charset.forName("windows-1251"), miniBatchSize, exampleLength,
                CharactersSets.getRussianCharacterSet(), new Random(1));

        int miniBatchNumber = 0;

        while (iter.hasNext()) {
            iter.next();
            miniBatchNumber++;
        }
        int expectedMinibatchNumber = (iter.getSize() / (miniBatchSize * exampleLength)) + 1;
        Assert.assertEquals(expectedMinibatchNumber,  miniBatchNumber);
    }

    @Test
    public void testEncoding() throws Exception {
        String s = "Лев Николаевич Толстой";
        char[] chars = s.toCharArray();

        File f = new File(CharacterIterator.class.getResource("/tolstoy_selected.txt").toURI());
        String line = Files.readAllLines(f.toPath(), Charset.forName("windows-1251")).iterator().next();

        for (int i = 0; i < chars.length; i++) {
            char c = chars[i];
            Assert.assertEquals(c, line.charAt(i));
        }
    }

}
