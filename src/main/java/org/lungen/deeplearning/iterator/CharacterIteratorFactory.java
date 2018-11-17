package org.lungen.deeplearning.iterator;

import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
import java.util.Random;

import static org.lungen.deeplearning.iterator.CharactersSets.*;

/**
 * Utility class for creating RussianCharacterIterator, e.g. from text file.
 */
public class CharacterIteratorFactory {

    /**
     *
     * DataSetIterator that does vectorization based on the text.
     * @param miniBatchSize Number of text segments in each training mini-batch
     * @param sequenceLength Number of characters in each text segment.
     */
    public static CharacterIterator getRussianFileIterator(
            String fileName, int miniBatchSize, int sequenceLength)
            throws Exception {

        String currentDir = System.getProperty("user.dir");
        String fileLocation = currentDir + "/src/main/resources/" + fileName;
        File f = new File(fileLocation);

        if (!f.exists()) {
            throw new IOException("Cannot find file: " + f.getCanonicalPath());
        } else {
            System.out.println("File found: " + f.getAbsolutePath());
        }

        // Which characters are allowed? Others will be removed
        char[] validCharacters = CharactersSets.createCharacterSet(CharactersSets.RUSSIAN, CharactersSets.LATIN, CharactersSets.NUMBERS, CharactersSets.PUNCTUATION);

        return new CharacterIterator(fileLocation, Charset.forName("windows-1251"),
                miniBatchSize, sequenceLength, validCharacters, new Random(12345));
    }

    /**
     *
     * DataSetIterator that does vectorization based on the text.
     * @param miniBatchSize Number of text segments in each training mini-batch
     * @param sequenceLength Number of characters in each text segment.
     */
    public static CharacterIterator getEnglishFileIterator(
            String fileName, int miniBatchSize, int sequenceLength)
            throws Exception {

        String currentDir = System.getProperty("user.dir");
        String fileLocation = currentDir + "/src/main/resources/" + fileName;
        File f = new File(fileLocation);

        if (!f.exists()) {
            throw new IOException("Cannot find file: " + f.getCanonicalPath());
        } else {
            System.out.println("File found: " + f.getAbsolutePath());
        }

        // Which characters are allowed? Others will be removed
        char[] validCharacters = CharactersSets.createCharacterSet(CharactersSets.LATIN, CharactersSets.NUMBERS, CharactersSets.PUNCTUATION);

        return new CharacterIterator(fileLocation, Charset.forName("windows-1251"),
                miniBatchSize, sequenceLength, validCharacters, new Random(12345));
    }


}
