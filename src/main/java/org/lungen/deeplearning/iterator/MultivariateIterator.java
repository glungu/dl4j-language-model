package org.lungen.deeplearning.iterator;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.MultiDataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.*;
import java.util.stream.IntStream;

import static org.lungen.deeplearning.iterator.CharactersSets.*;

/**
 * MultivariateIterator
 * A multi dataset iterator for multivariate input.
 * Input consists of 2 datasets - one for encoding character sequence,
 * another - for the rest of non-sequential features.
 * Characters in the character sequence represented as one-hot vectors.
 *
 * Each label is a numerical feature.
 *
 * @author lungen.tech@gmail.com
 */
public class MultivariateIterator implements MultiDataSetIterator {

    private static final Logger log = LoggerFactory.getLogger("autoencoder.iterator");

    private Random rng = new Random(7);

    private int miniBatchSize;
    private int charSequenceColumn;
    private int charSequenceMaxLength;
    private char[] charDictionary;

    private Map<Character,Integer> mapCharToIndex;

    private CSVParser csvParser;
    private List<char[]> charSequences;
    private ArrayList<long[]> nonSequenceData;
    private List<Long> labelData;

    // offsets for the start of each example
    private LinkedList<Integer> exampleStartOffsets = new LinkedList<>();


    public MultivariateIterator() {
    }

    public MultivariateIterator(File csvFile,
                                int miniBatchSize,
                                String charSequenceColumnName,
                                char[] charsValid) {

        if (!csvFile.isFile()) {
            throw new IllegalStateException("File does not exist: " + csvFile);
        }
        if (miniBatchSize <= 0) {
            throw new IllegalArgumentException("Invalid miniBatchSize (must be >0)");
        }
        this.miniBatchSize = miniBatchSize;
        this.charDictionary = charsValid;

        // valid characters mapping for use in vectorization
        this.mapCharToIndex = new HashMap<>();
        for (int i = 0; i < charDictionary.length; i++) {
            mapCharToIndex.put(charDictionary[i], i);
        }

        // load file
        csvParser = new CSVParser(csvFile, true);
        List<List<String>> parsedLines = csvParser.getParsedLines();

        // determine character sequence max length
        charSequenceColumn = csvParser.getHeaderIndex(charSequenceColumnName);
        charSequences = new ArrayList<>();
        nonSequenceData = new ArrayList<>();
        labelData = new ArrayList<>();
        for (List<String> parsedLine : parsedLines) {
            // sequence
            char[] chars = parsedLine.get(charSequenceColumn).toCharArray();
            char[] charsCleaned = cleanInvalidCharacters(chars);
            charSequences.add(charsCleaned);
            // non-sequence
            long[] numericValues = IntStream.range(0, csvParser.headerNames.length).filter(i -> {
                String headerName = csvParser.headerNames[i];
                return !headerName.equals(charSequenceColumnName) && i != csvParser.headerNames.length - 1;
            }).mapToLong(i -> {
                String headerName = csvParser.headerNames[i];
                String value = parsedLine.get(csvParser.getHeaderIndex(headerName));
                return value.equals("") ? -1 : Long.valueOf(value);
            }).toArray();
            nonSequenceData.add(numericValues);
            // label
            String labelValue = parsedLine.get(csvParser.headerNames.length - 1);
            labelData.add(labelValue.equals("") ? -1 : Long.valueOf(labelValue));
        }
        IntSummaryStatistics summary = charSequences.stream().mapToInt(chars -> chars.length).summaryStatistics();
        charSequenceMaxLength = summary.getMax();
        log.info("### Character Sequence Length: " + summary.toString());

        // non-sequence data


        initializeOffsets();
        log.info("### Number of batches per epoch: " + exampleStartOffsets.size());
    }

    private void initializeOffsets() {
        //This defines the order in which parts of the file are fetched
        int nMinibatchesPerEpoch = (int) Math.ceil(csvParser.getParsedLines().size() / (double) miniBatchSize);

        for (int i = 0; i < nMinibatchesPerEpoch; i++) {
            exampleStartOffsets.add(i * miniBatchSize);
        }
        Collections.shuffle(exampleStartOffsets, rng);
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
    public void setPreProcessor(MultiDataSetPreProcessor preProcessor) {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public MultiDataSetPreProcessor getPreProcessor() {
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

    }

    @Override
    public boolean hasNext() {
        return false;
    }

    @Override
    public MultiDataSet next() {
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
    public MultiDataSet next(int batchSize) {
        if (exampleStartOffsets.size() == 0) {
            throw new NoSuchElementException();
        }

        int currMinibatchSize = Math.min(batchSize, exampleStartOffsets.size());

        //Allocate space:
        //Note the order here:
        // dimension 0 = number of examples in minibatch
        // dimension 1 = size of each vector (i.e., number of characters)
        // dimension 2 = length of each time series/example
        //Why 'f' order here? See http://deeplearning4j.org/usingrnns.html#data section "Alternative: Implementing a custom DataSetIterator"
        INDArray sequenceInput = Nd4j.create(new int[] {currMinibatchSize, charDictionary.length, charSequenceMaxLength}, 'f');
        INDArray nonSequenceInput = Nd4j.create(new int[] {currMinibatchSize, nonSequenceData.get(0).length}, 'f');
        INDArray labels = Nd4j.create(new int[] {currMinibatchSize, 1}, 'f');

        for (int i = 0; i < currMinibatchSize; i++) {
            // sequence
            int sequenceIndex = exampleStartOffsets.removeFirst();
            char[] charSequence = charSequences.get(sequenceIndex);
            int c = 0;
            for (int j = 0; j < charSequenceMaxLength; j++, c++) {
                int charIndex = mapCharToIndex.get(charSequence[j]);        //Next character to predict
                sequenceInput.putScalar(new int[] {i, charIndex, c}, 1.0);
            }
            // non-sequence
            long[] numeric = nonSequenceData.get(sequenceIndex);
            nonSequenceInput.put(i, Nd4j.create(numeric));
            // labels
            labels.putScalar(i, labelData.get(sequenceIndex));
        }
        return new org.nd4j.linalg.dataset.MultiDataSet(
                new INDArray[] {sequenceInput, nonSequenceInput},
                new INDArray[] {labels});
    }


    public static void main(String[] args) {
        String dir = "C:/DATA/Projects/DataSets/Jnetx_Bugzilla";
        String filePath = dir + "/bugzilla-jnetx-processed-final.csv";
        MultivariateIterator iterator = new MultivariateIterator(new File(filePath),
                1, "description",
                createCharacterSet(LATIN, RUSSIAN, NUMBERS, PUNCTUATION, SPECIAL));
        MultiDataSet batch = iterator.next();

    }
}
