package org.lungen.deeplearning.iterator;

import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.Random;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * A multi dataset iterator for autoencoder.
 * Input consists of same datasets - one for encoding character sequence,
 * another - for decoding. Characters represented as one-hot vectors.
 *
 * Labels are same character sequences but shifted to the left, i.e.
 * representing next character for each of the characters in input sequence.
 *
 * @author lungen.tech@gmail.com
 */
public class AutoEncoderCharacterIterator implements MultiDataSetIterator {

    private static final Logger log = LoggerFactory.getLogger("autoencoder.iterator");

    //Valid characters
    protected char[] validCharacters;
    //Maps each character to an index ind the input/output
    protected Map<Character,Integer> charToIdxMap;
    //All characters of the input file (after filtering to only those that are valid
    private char[] fileCharacters;
    //Length of each example/minibatch (number of characters)
    protected int exampleLength;
    //Size of each minibatch (number of examples)
    protected int miniBatchSize;
    protected Random rng;
    //Offsets for the start of each example
    private LinkedList<Integer> exampleStartOffsets = new LinkedList<>();

    public AutoEncoderCharacterIterator() {
    }

    public AutoEncoderCharacterIterator(File textFile, int miniBatchSize, int exampleLength, char[] validCharacters) {
        this(textFile.getAbsolutePath(), Charset.forName("utf-8"), miniBatchSize, exampleLength, validCharacters);
    }

    /**
     * @param textFilePath Path to text file to use for generating samples
     * @param textFileEncoding Encoding of the text file. Can try Charset.defaultCharset()
     * @param miniBatchSize Number of examples per mini-batch
     * @param exampleLength Number of characters in each input/output vector
     * @param validCharacters Character array of valid characters. Characters not present in this array will be removed
     * @throws IOException If text file cannot  be loaded
     */
    public AutoEncoderCharacterIterator(String textFilePath, Charset textFileEncoding, int miniBatchSize, int exampleLength,
                                        char[] validCharacters) {
        this(textFilePath,textFileEncoding,miniBatchSize,exampleLength,validCharacters,null);
    }
    /**
     * @param textFilePath Path to text file to use for generating samples
     * @param textFileEncoding Encoding of the text file. Can try Charset.defaultCharset()
     * @param miniBatchSize Number of examples per mini-batch
     * @param exampleLength Number of characters in each input/output vector
     * @param validCharacters Character array of valid characters. Characters not present in this array will be removed
     * @param commentChars if non-null, lines starting with this string are skipped.
     * @throws IOException If text file cannot  be loaded
     */
    public AutoEncoderCharacterIterator(String textFilePath, Charset textFileEncoding, int miniBatchSize, int exampleLength,
                                        char[] validCharacters, String commentChars)
    {
        if (!new File(textFilePath).exists()) {
            throw new IllegalStateException("Could not access file (does not exist): " + textFilePath);
        }
        if (miniBatchSize <= 0) {
            throw new IllegalArgumentException("Invalid miniBatchSize (must be >0)");
        }
        this.validCharacters = validCharacters;
        this.exampleLength = exampleLength;
        this.miniBatchSize = miniBatchSize;

        //Store valid characters is a map for later use in vectorization
        charToIdxMap = new HashMap<>();
        for (int i = 0; i < validCharacters.length; i++) {
            charToIdxMap.put(validCharacters[i], i);
        }

        //Load file and convert contents to a char[]
        boolean newLineValid = charToIdxMap.containsKey('\n');
        List<String> lines = null;
        try {
            lines = Files.readAllLines(new File(textFilePath).toPath(),textFileEncoding);
            log.info("File read. Number of lines: " + lines.size());
        } catch (IOException e) {
            log.error("Error reading file: " + textFilePath, e);
            throw new IllegalStateException("Cannot read from file");
        }
        if (commentChars != null) {
            List<String> withoutComments = new ArrayList<>();
            for (String line : lines) {
                if (!line.startsWith(commentChars)) {
                    withoutComments.add(line);
                }
            }
            lines = withoutComments;
        }
        // add lines.size() to account for newline characters at end of each line
        int maxSize = lines.size();
        for (String s : lines) {
            maxSize += s.length();
        }

        log.info("Total characters: " + maxSize);
        char[] characters = new char[maxSize];
        int currIdx = 0;
        for (String s : lines) {
            char[] thisLine = s.toCharArray();
            for (char aThisLine : thisLine) {
                if (!charToIdxMap.containsKey(aThisLine)) {
                    continue;
                }
                characters[currIdx++] = aThisLine;
            }
            if (newLineValid) {
                characters[currIdx++] = '\n';
            }
        }

        log.info("Total valid characters: " + currIdx);
        if (currIdx == characters.length) {
            fileCharacters = characters;
        } else {
            fileCharacters = Arrays.copyOfRange(characters, 0, currIdx);
        }
        if (exampleLength >= fileCharacters.length) {
            throw new IllegalArgumentException("exampleLength=" + exampleLength
                    + " cannot exceed number of valid characters in file (" + fileCharacters.length + ")");
        }

        int nRemoved = maxSize - fileCharacters.length;
        System.out.println("### File loaded, valid characters: " + fileCharacters.length + ", "
                + "total characters: " + maxSize + ", removed: " + nRemoved);

        initializeOffsets();
        log.info("### Number of batches per epoch: " + ((exampleStartOffsets.size() / miniBatchSize) + 1));
    }

    public int getExampleLength() {
        return exampleLength;
    }

    public int getMiniBatchSize() {
        return miniBatchSize;
    }

    public char convertIndexToCharacter(int idx ){
        return validCharacters[idx];
    }

    public int convertCharacterToIndex( char c ){
        return charToIdxMap.get(c);
    }

    public char getRandomCharacter() {
        return validCharacters[(int) (rng.nextDouble() * validCharacters.length)];
    }

    public boolean hasNext() {
        return exampleStartOffsets.size() > 0;
    }

    public MultiDataSet next() {
        return next(miniBatchSize);
    }

    public MultiDataSet next(int num) {
        if (exampleStartOffsets.size() == 0) {
            throw new NoSuchElementException();
        }

        int currMinibatchSize = Math.min(num, exampleStartOffsets.size());
        //Allocate space:
        //Note the order here:
        // dimension 0 = number of examples in minibatch
        // dimension 1 = size of each vector (i.e., number of characters)
        // dimension 2 = length of each time series/example
        //Why 'f' order here? See http://deeplearning4j.org/usingrnns.html#data section "Alternative: Implementing a custom DataSetIterator"
        INDArray input = Nd4j.create(new int[]{currMinibatchSize,validCharacters.length,exampleLength}, 'f');
        INDArray decode = Nd4j.create(new int[]{currMinibatchSize,validCharacters.length,exampleLength}, 'f');
        INDArray labels = Nd4j.create(new int[]{currMinibatchSize,validCharacters.length,exampleLength}, 'f');

        for( int i=0; i<currMinibatchSize; i++ ){
            int startIdx = exampleStartOffsets.removeFirst();
            int endIdx = startIdx + exampleLength;
            int currCharIdx = charToIdxMap.get(fileCharacters[startIdx]);	//Current input
            int c=0;
            for( int j=startIdx+1; j<endIdx; j++, c++ ){
                int nextCharIdx = charToIdxMap.get(fileCharacters[j]);		//Next character to predict
                input.putScalar(new int[]{i,currCharIdx,c}, 1.0);
                decode.putScalar(new int[]{i,currCharIdx,c}, 1.0);
                labels.putScalar(new int[]{i,nextCharIdx,c}, 1.0);
                currCharIdx = nextCharIdx;
            }
        }
        return new org.nd4j.linalg.dataset.MultiDataSet(
                new INDArray[] {input, decode},
                new INDArray[] {labels});
    }

    public int totalExamples() {
        return (fileCharacters.length-1) / miniBatchSize - 2;
    }

    public int getDictionarySize() {
        return validCharacters.length;
    }

    public void reset() {
        exampleStartOffsets.clear();
        initializeOffsets();
    }

    protected void initializeOffsets() {
        rng = new Random(12345);
        //This defines the order in which parts of the file are fetched
        int nMinibatchesPerEpoch = (fileCharacters.length - 1) / exampleLength - 2;   //-2: for end index, and for partial example
        for (int i = 0; i < nMinibatchesPerEpoch; i++) {
            exampleStartOffsets.add(i * exampleLength);
        }
        Collections.shuffle(exampleStartOffsets, rng);
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

    @Override
    public void setPreProcessor(MultiDataSetPreProcessor multiDataSetPreProcessor) {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public MultiDataSetPreProcessor getPreProcessor() {
        throw new UnsupportedOperationException("Not implemented");
    }

    public List<String> getLabels() {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public void remove() {
        throw new UnsupportedOperationException();
    }

    public String getRandomExample() {
        StringBuilder str = new StringBuilder();
        Integer currentExampleStart = exampleStartOffsets.getFirst();
        for (int i = 0; i < exampleLength; i++) {
            str.append(fileCharacters[currentExampleStart + i]);
        }
        return str.toString();
    }

    public String recordToString(INDArray array) {
        String[] result = new String[(int) array.size(0)];
        for (int i = 0; i < array.size(0); i++) {
            StringBuilder sb = new StringBuilder();
            for (int j = 0; j < array.size(2); j++) {
                INDArray charOneHot = array.get(
                        NDArrayIndex.point(i),
                        NDArrayIndex.all(),
                        NDArrayIndex.point(j));
                int argmax = charOneHot.argMax(0).getInt(0);
                sb.append(convertIndexToCharacter(argmax));
            }
            result[i] = sb.toString();
        }
        return result[0];
    }

}
