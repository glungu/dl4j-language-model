package org.lungen.deeplearning.iterator;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.IntSummaryStatistics;
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
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class CharacterSequenceClassifierIterator implements DataSetIterator {

    private static final Logger log = LoggerFactory.getLogger("autoencoder.iterator");

    private Random rng = new Random(7);

    private int miniBatchSize;
    private int charSequenceColumn;
    private int charSequenceMaxLength;
    private char[] charDictionary;

    private Map<Character, Integer> mapCharToIndex;

    private CSVParser csvParser;
    private List<char[]> charSequences;
    private List<Long> labels;

    // offsets for the start of each example
    private LinkedList<Integer> miniBatchStartOffsets = new LinkedList<>();
    private int numOuputClasses;


    public CharacterSequenceClassifierIterator() {
    }

    public CharacterSequenceClassifierIterator(File csvFile,
                                               char[] charsValid,
                                               int numOuputClasses,
                                               int miniBatchSize) {

        if (!csvFile.isFile()) {
            throw new IllegalStateException("File does not exist: " + csvFile);
        }
        if (miniBatchSize <= 0) {
            throw new IllegalArgumentException("Invalid miniBatchSize (must be >0)");
        }
        this.miniBatchSize = miniBatchSize;
        this.charDictionary = charsValid;
        this.numOuputClasses = numOuputClasses;

        // valid characters mapping for use in vectorization
        this.mapCharToIndex = new HashMap<>();
        for (int i = 0; i < charDictionary.length; i++) {
            mapCharToIndex.put(charDictionary[i], i);
        }

        // load file
        csvParser = new CSVParser(csvFile, true);
        List<List<String>> parsedLines = csvParser.getParsedLines();

        // determine character sequence max length
        charSequenceColumn = 0;
        charSequences = new ArrayList<>();
        labels = new ArrayList<>();
        for (List<String> parsedLine : parsedLines) {
            // sequences
            char[] chars = parsedLine.get(charSequenceColumn).toCharArray();
            char[] charsCleaned = cleanInvalidCharacters(chars);
            charSequences.add(charsCleaned);
            // labels
            String labelValue = parsedLine.get(csvParser.headerNames.length - 1);
            labels.add(labelValue.equals("") ? -1 : Long.valueOf(labelValue));
        }
        IntSummaryStatistics summary = charSequences.stream().mapToInt(chars -> chars.length).summaryStatistics();
        charSequenceMaxLength = summary.getMax();
        log.info("### Character Sequence Summary: \n\t" + summary.toString());

        IntSummaryStatistics summaryLabels = labels.stream().mapToInt(Long::intValue).summaryStatistics();
        log.info("### Labels Summary: \n\t" + summaryLabels.toString());

        initializeOffsets();
        log.info("### Number of batches per epoch: " + miniBatchStartOffsets.size());
    }

    private void initializeOffsets() {
        //This defines the order in which parts of the file are fetched
        int nMinibatchesPerEpoch = (int) Math.ceil(csvParser.getParsedLines().size() / (double) miniBatchSize);

        for (int i = 0; i < nMinibatchesPerEpoch; i++) {
            miniBatchStartOffsets.add(i * miniBatchSize);
        }
        Collections.shuffle(miniBatchStartOffsets, rng);
    }

    private char[] cleanInvalidCharacters(char[] input) {
        char[] result = new char[input.length];
        int index = 0;
        for (char c : input) {
            if (!mapCharToIndex.containsKey(c)) {
                continue;
            }
            result[index++] = c;
        }
        return Arrays.copyOfRange(result, 0, index);
    }


    @Override
    public void setPreProcessor(DataSetPreProcessor preProcessor) {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public DataSetPreProcessor getPreProcessor() {
        return null;
    }

    @Override
    public List<String> getLabels() {
        return null;
    }

    @Override
    public boolean resetSupported() {
        return true;
    }

    @Override
    public boolean asyncSupported() {
        return true;
    }

    @Override
    public void reset() {
        miniBatchStartOffsets.clear();
        initializeOffsets();
    }

    @Override
    public int batch() {
        return miniBatchSize;
    }

    public int getCurrentPosition() {
        return !miniBatchStartOffsets.isEmpty() ? this.miniBatchStartOffsets.getFirst() : -1;
    }

    public char[] getSequence(int index) {
        return index >= 0 && index < charSequences.size() ? charSequences.get(index) : null;
    }

    public void get() {

    }

    @Override
    public boolean hasNext() {
        return miniBatchStartOffsets.size() > 0;
    }

    @Override
    public DataSet next() {
        return next(miniBatchSize);
    }

    /**
     * Return next minibatch of data,
     * i.e. minibatchSize number of maxLength sequences from the original file.
     * Sequences have variable length, so they change to maxLength and masked.
     * Since each character is represented as one-hot vector, then we have an array
     * of shape BatchSize x V x MaxLength (where V is number of valid characters).
     *
     * @param batchSize The minibatch size
     * @return DataSet for next minibatch
     */
    @Override
    public DataSet next(int batchSize) {
        if (miniBatchStartOffsets.size() == 0) {
            throw new NoSuchElementException();
        }

        int currMinibatchSize = Math.min(batchSize, miniBatchStartOffsets.size());

//        //Allocate space:
//        //Note the order here:
//        // dimension 0 = number of examples in minibatch
//        // dimension 1 = size of each vector (i.e., number of characters)
//        // dimension 2 = length of each time series/example
//        //Why 'f' order here? See http://deeplearning4j.org/usingrnns.html#data section "Alternative: Implementing a custom DataSetIterator"
//        INDArray sequenceInput = Nd4j.create(new int[]{currMinibatchSize, charDictionary.length, charSequenceMaxLength}, 'f');
//        INDArray nonSequenceInput = Nd4j.create(new int[]{currMinibatchSize, nonSequenceSize}, 'f');
//        INDArray labels = Nd4j.create(new int[]{currMinibatchSize, 1}, 'f');

        // data
        INDArray sequenceInput = Nd4j.zeros(currMinibatchSize, charDictionary.length, charSequenceMaxLength);
        INDArray labels = Nd4j.zeros(currMinibatchSize, 1);

        // masks
        INDArray sequenceInputMask = Nd4j.zeros(currMinibatchSize, charSequenceMaxLength);
        INDArray labelsMask = Nd4j.ones(currMinibatchSize, 1);

        for (int i = 0; i < currMinibatchSize; i++) {
            // sequence
            int index = miniBatchStartOffsets.removeFirst();
            char[] charSequence = charSequences.get(index);
            int c = 0;
            for (int j = 0; j < charSequence.length; j++, c++) {
                int charIndex = mapCharToIndex.get(charSequence[j]);
                sequenceInput.putScalar(new int[]{i, charIndex, c}, 1.0);
            }
            // labels
            labels.putScalar(i, this.labels.get(index));

            // mask
            sequenceInputMask.put(
                    new INDArrayIndex[] {
                            NDArrayIndex.point(i),
                            NDArrayIndex.interval(0, charSequence.length)
                    },
                    Nd4j.ones(charSequence.length));

        }
        return new org.nd4j.linalg.dataset.DataSet(
                sequenceInput, labels,
                sequenceInputMask, labelsMask);
    }

    @Override
    public int inputColumns() {
        return getNumSequenceFeatures();
    }

    @Override
    public int totalOutcomes() {
        return numOuputClasses;
    }

    public int charToIndex(char c) {
        return this.mapCharToIndex.get(c);
    }

    public char indexToChar(int i) {
        return this.charDictionary[i];
    }

    public int getNumSequenceFeatures() {
        return charDictionary.length;
    }

}

