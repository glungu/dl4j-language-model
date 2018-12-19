package org.lungen.deeplearning.iterator;

import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.Random;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** A simple DataSetIterator for use in the GravesLSTMCharModellingExample.
 * Given a text file and a few options, generate feature vectors and labelData for training,
 * where we want to predict the next character in the sequence.<br>
 * This is done by randomly choosing a position in the text file, at offsets of 0, exampleLength, 2*exampleLength, etc
 * to start each sequence. Then we convert each character to an index, i.e., a one-hot vector.
 * Then the character 'a' becomes [1,0,0,0,...], 'b' becomes [0,1,0,0,...], etc
 *
 * Feature vectors and labelData are both one-hot vectors of same length
 * @author Alex Black
 */
public class CharacterIterator implements DataSetIterator {

    private static final Logger log = LoggerFactory.getLogger("iterator.character");

    //Valid characters
    private char[] validCharacters;
    //Maps each character to an index ind the input/output
    private Map<Character,Integer> charToIdxMap;
    //All characters of the input file (after filtering to only those that are valid
    private char[] fileCharacters;
    //Length of each example/minibatch (number of characters)
    private int exampleLength;
    //Size of each minibatch (number of examples)
    private int miniBatchSize;

    private Random rng;

    //Offsets for the start of each example
    private LinkedList<Integer> exampleStartOffsets = new LinkedList<>();

    public CharacterIterator(File textFile,
                             int miniBatchSize,
                             int exampleLength,
                             char[] validCharacters) throws IOException {
        this(textFile.getAbsolutePath(), Charset.forName("utf-8"), miniBatchSize, exampleLength, validCharacters, new Random(7));
    }

        /**
         * @param textFilePath Path to text file to use for generating samples
         * @param textFileEncoding Encoding of the text file. Can try Charset.defaultCharset()
         * @param miniBatchSize Number of examples per mini-batch
         * @param exampleLength Number of characters in each input/output vector
         * @param validCharacters Character array of valid characters. Characters not present in this array will be removed
         * @param rng Random number generator, for repeatability if required
         * @throws IOException If text file cannot  be loaded
         */
    public CharacterIterator(String textFilePath,
                             Charset textFileEncoding,
                             int miniBatchSize,
                             int exampleLength,
                             char[] validCharacters,
                             Random rng) throws IOException {

        if (!new File(textFilePath).exists()) {
            throw new IOException("Could not access file (does not exist): " + textFilePath);
        }
        if (miniBatchSize <= 0) {
            throw new IllegalArgumentException("Invalid miniBatchSize (must be >0)");
        }
        this.validCharacters = validCharacters;
        this.exampleLength = exampleLength;
        this.miniBatchSize = miniBatchSize;
        this.rng = rng;

        // Store valid characters is a map for later use in vectorization:
        // valid char -> index in validCharacters array
        charToIdxMap = new HashMap<>();
        for (int i = 0; i < validCharacters.length; i++) {
            charToIdxMap.put(validCharacters[i], i);
        }

        // Load file and count size
        boolean newLineValid = charToIdxMap.containsKey('\n');
        List<String> lines = Files.readAllLines(new File(textFilePath).toPath(),textFileEncoding);
        int maxSize = lines.size();	//add lines.size() to account for newline characters at end of each line
        for (String s : lines) {
            maxSize += s.length();
        }

        // Convert contents to a char[] (only valid characters)
        char[] characters = new char[maxSize];
        int currIdx = 0;
        for (String s : lines) {
            char[] lineChars = s.toCharArray();
            for (char c : lineChars) {
                if (!charToIdxMap.containsKey(c)) {
                    continue;
                }
                characters[currIdx++] = c;
            }
            if (newLineValid) {
                characters[currIdx++] = '\n';
            }
        }

        if (currIdx == characters.length) {
            fileCharacters = characters;
        } else {
            fileCharacters = Arrays.copyOfRange(characters, 0, currIdx);
        }

        // Check total length vs example length
        if (exampleLength >= fileCharacters.length) {
            throw new IllegalArgumentException("exampleLength=" + exampleLength
                    + " cannot exceed number of valid characters in file "
                    + "(" + fileCharacters.length + ")");
        }

        int nRemoved = maxSize - fileCharacters.length;
        log.info("Loaded and converted file: \n"
                + "\t Valid characters: " + fileCharacters.length + "\n"
                + "\t Total characters: " + maxSize + "\n"
                + "\t Removed characters: " + nRemoved);

        // Divide fileCharacters into exampleLength chunks and shuffle
        int totalExamples = initializeOffsets();
        log.info("Minibatches per epoch: " + (int) Math.ceil(totalExamples / (double)miniBatchSize));
    }

    public char convertIndexToCharacter( int idx ){
        return validCharacters[idx];
    }

    public int convertCharacterToIndex( char c ){
        return charToIdxMap.get(c);
    }

    public char getRandomCharacter(){
        return validCharacters[(int) (rng.nextDouble()*validCharacters.length)];
    }

    public boolean hasNext() {
        return exampleStartOffsets.size() > 0;
    }

    public DataSet next() {
        return next(miniBatchSize);
    }

    /**
     * Return next minibatch of data,
     * i.e. minibatchSize number of exampleLength sequences from the original file.
     * If minibatchSize = 32 and exampleLength = 1000, then it is
     * 32 sequences of 1000 characters.
     * Since each character is represented as one-hot vector, then we have
     * an array of shape 32 x V x 1000 (where H is number of valid characters).
     *
     * @param batchSize The minibatch size
     * @return DataSet for next minibatch
     */
    public DataSet next(int batchSize) {
        if (exampleStartOffsets.size() == 0) {
            throw new NoSuchElementException();
        }

        int currMinibatchSize = Math.min(batchSize, exampleStartOffsets.size());
        // Allocate space:
        // Note the order here:
        //  dimension 0 = number of examples in minibatch
        //  dimension 1 = size of each vector (i.e., number of characters)
        //  dimension 2 = length of each time series/example
        int[] shape = {currMinibatchSize, validCharacters.length, exampleLength};

        // Why 'f' order here? See http://deeplearning4j.org/usingrnns.html#data section "Alternative: Implementing a custom DataSetIterator"
        INDArray input = Nd4j.create(shape, 'f');
        INDArray labels = Nd4j.create(shape, 'f');

        // Fill input and labelData with one-hot data
        // Use charToIdxMap to determine index of 1.0 (to represent as one-hot)
        for (int i = 0; i < currMinibatchSize; i++) {
            int startIdx = exampleStartOffsets.removeFirst();
            int endIdx = startIdx + exampleLength;
            int currCharIdx = charToIdxMap.get(fileCharacters[startIdx]);    //Current input
            int c = 0;
            for (int j = startIdx + 1; j < endIdx; j++, c++) {
                int nextCharIdx = charToIdxMap.get(fileCharacters[j]);        //Next character to predict
                input.putScalar(new int[]{i, currCharIdx, c}, 1.0);
                labels.putScalar(new int[]{i, nextCharIdx, c}, 1.0);
                currCharIdx = nextCharIdx;
            }
        }

        return new DataSet(input, labels);
    }

    public int totalExamples() {
        return (fileCharacters.length-1) / miniBatchSize - 2;
    }

    public int inputColumns() {
        return validCharacters.length;
    }

    public int totalOutcomes() {
        return validCharacters.length;
    }

    public void reset() {
        exampleStartOffsets.clear();
        initializeOffsets();
    }

    private int initializeOffsets() {
        // This defines the order in which parts of the file are fetched
        int nMinibatchesPerEpoch = (fileCharacters.length - 1) / exampleLength - 2;   //-2: for end index, and for partial example
        for (int i = 0; i < nMinibatchesPerEpoch; i++) {
            exampleStartOffsets.add(i * exampleLength);
        }
        Collections.shuffle(exampleStartOffsets, rng);
        return nMinibatchesPerEpoch;
    }

    public boolean resetSupported() {
        return true;
    }

    @Override
    public boolean asyncSupported() {
        return true;
    }

    public int batch() {
        return miniBatchSize;
    }

    public int cursor() {
        return totalExamples() - exampleStartOffsets.size();
    }

    public int numExamples() {
        return totalExamples();
    }

    public int getExampleLength() {
        return exampleLength;
    }

    public void setPreProcessor(DataSetPreProcessor preProcessor) {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public DataSetPreProcessor getPreProcessor() {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public List<String> getLabels() {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public void remove() {
        throw new UnsupportedOperationException();
    }

    public int getSize() {
        return fileCharacters != null ? fileCharacters.length : -1;
    }

    public char[] getValidCharacters() {
        return validCharacters;
    }
}
