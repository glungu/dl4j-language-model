package org.lungen.deeplearning.iterator;

import org.junit.Assert;
import org.junit.Test;

import java.io.File;
import java.nio.charset.Charset;
import java.nio.file.Files;

/**
 * Unit-test for CharacterIteratorFactory
 */
public class CharacterIteratorFactoryTest {

    @Test
    public void testSize() throws Exception {
        int miniBatchSize = 16;
        int exampleLength = 500;
        CharacterIterator iter = CharacterIteratorFactory.getRussianFileIterator(
                "tolstoy_selected.txt", miniBatchSize, exampleLength);

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

        String currentDir = System.getProperty("user.dir");
        String fileLocation = currentDir + "/src/main/resources/tolstoy_selected.txt";
        File f = new File(fileLocation);
        String line = Files.readAllLines(f.toPath(), Charset.forName("windows-1251")).iterator().next();

        for (int i = 0; i < chars.length; i++) {
            char c = chars[i];
            Assert.assertEquals(c, line.charAt(i));
        }
    }

}
