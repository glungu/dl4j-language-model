package org.lungen.deeplearning.iterator;

import org.lungen.data.bugzilla.BugzillaResolveTime;
import org.lungen.data.bugzilla.CSVParser;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.*;
import java.util.function.Function;
import java.util.stream.Collectors;

/**
 * CharacterSequenceValuePredictorIterator
 *
 * @author lungen.tech@gmail.com
 */
public class StringClassifierIterator implements DataSetIterator {

    private static final Logger log = LoggerFactory.getLogger("iterator.predictor");

    private Random rng = new Random(7);

    private int miniBatchSize;
    private int charSequenceColumn;
    private int charSequenceMaxLength;
    private char[] charDictionary;

    private Map<Character, Integer> mapCharToIndex;

    private CSVParser csvParser;
    private List<char[]> charSequences;
    private List<Integer> labels;
    private List<BugzillaResolveTime> labelClasses;
    private int numLabelClasses;

    // offsets for the start of each example
    private LinkedList<Integer> miniBatchStartOffsets = new LinkedList<>();


    public StringClassifierIterator(File csvFile,
                                    char[] charsValid,
                                    int charSequenceMaxLength,
                                    int miniBatchSize) {

        if (!csvFile.isFile()) {
            throw new IllegalStateException("File does not exist: " + csvFile);
        }
        if (miniBatchSize <= 0) {
            throw new IllegalArgumentException("Invalid miniBatchSize (must be >0)");
        }
        this.charSequenceMaxLength = charSequenceMaxLength;
        this.miniBatchSize = miniBatchSize;
        this.charDictionary = charsValid;

        // valid characters mapping for use in vectorization
        this.mapCharToIndex = new HashMap<>();
        for (int i = 0; i < charDictionary.length; i++) {
            mapCharToIndex.put(charDictionary[i], i);
        }

        // load file
        csvParser = new CSVParser(csvFile, false);
        List<List<String>> parsedLines = csvParser.getParsedLines();

        // process sequences and labels
        charSequenceColumn = 0;
        charSequences = new ArrayList<>();
        labels = new ArrayList<>();

        labelClasses = new ArrayList<>();
        numLabelClasses = BugzillaResolveTime.values().length;

        for (List<String> parsedLine : parsedLines) {
            if (parsedLine.size() != 2) {
                throw new IllegalStateException("Wrong line: " + parsedLine);
            }
            // sequences
            char[] chars = parsedLine.get(charSequenceColumn).toCharArray();
            char[] charsCleaned = cleanInvalidCharacters(chars);
            charSequences.add(charsCleaned);
            // labels
            String labelStr = parsedLine.get(parsedLine.size() - 1).trim();
            int labelInt = Integer.valueOf(labelStr);
            labels.add(labelInt);
            //label classes
            labelClasses.add(BugzillaResolveTime.values()[labelInt]);
        }
        IntSummaryStatistics summary = charSequences.stream().mapToInt(chars -> chars.length).summaryStatistics();
        IntSummaryStatistics summaryLabels = labels.stream().mapToInt(Integer::intValue).summaryStatistics();
        Map<BugzillaResolveTime, Long> labelClassesDistr = labelClasses.stream().collect(Collectors.groupingBy(Function.identity(), Collectors.counting()));
        StringBuilder summaryLabelClasses = new StringBuilder();
        labelClassesDistr.entrySet().stream()
                .sorted(Comparator.comparing(Map.Entry::getKey))
                .forEach(e -> summaryLabelClasses.append(e.getKey()).append(":").append(e.getValue()).append(","));

        // initialize minibatch offsets
        initializeOffsets();

        String summaryMsg = "Character sequence iterator summary:\n" +
                "\t Sequences:     " + summary.toString() + "\n" +
                "\t Labels:        " + summaryLabels.toString() + "\n" +
                "\t Label classes: " + summaryLabelClasses.toString() + "\n" +
                "\t Batches/epoch: " + miniBatchStartOffsets.size();
        log.info(summaryMsg);
    }

    private void initializeOffsets() {
        //This defines the order in which parts of the file are fetched
        int nMinibatchesPerEpoch = (int) Math.ceil(csvParser.getParsedLines().size() / (double) miniBatchSize);

        for (int i = 0; i < nMinibatchesPerEpoch; i++) {
            miniBatchStartOffsets.add(i * miniBatchSize);
        }
        // data in file already shuffled
        // Collections.shuffle(miniBatchStartOffsets, rng);
    }

    private char[] cleanInvalidCharacters(char[] input) {
        char[] result = new char[input.length];
        int index = 0;
        for (char c : input) {
            // remove all leading quotes
            if (index == 0 && c == '"') {
                continue;
            }
            // remove multiple quotes
            if (c == '"' && index >= 1 && result[index - 1] == '"') {
                continue;
            }
            if (!mapCharToIndex.containsKey(c)) {
                continue;
            }
            result[index++] = c;
            // max length
            if (index >= charSequenceMaxLength) {
                break;
            }
        }
        // remove all trailing quotes (can be one at this point)
        if (result[index-1] == '"') {
            index--;
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

    public int getTotalSequences() {
        return charSequences.size();
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

        int indexStart = miniBatchStartOffsets.removeFirst();
        int currMinibatchSize = Math.min(batchSize, charSequences.size() - indexStart);

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
        INDArray sequenceInput = Nd4j.zeros(new int[]{currMinibatchSize, charDictionary.length, charSequenceMaxLength}, 'f');
        INDArray labels = Nd4j.zeros(new int[]{currMinibatchSize, numLabelClasses}, 'f');

        // masks
        INDArray sequenceInputMask = Nd4j.zeros(new int[]{currMinibatchSize, charSequenceMaxLength}, 'f');
        INDArray labelsMask = Nd4j.zeros(new int[]{currMinibatchSize, numLabelClasses}, 'f');
        labelsMask.put(
                new INDArrayIndex[]{NDArrayIndex.all(), NDArrayIndex.all()},
                Nd4j.ones(currMinibatchSize, numLabelClasses));

        for (int i = 0; i < currMinibatchSize; i++) {
            // sequence
            char[] charSequence = charSequences.get(indexStart + i);
            int c = 0;
            int len = Math.min(charSequence.length, charSequenceMaxLength);
            for (int j = 0; j < len; j++, c++) {
                try {
                    int charIndex = mapCharToIndex.get(charSequence[j]);
                    sequenceInput.putScalar(new int[]{i, charIndex, c}, 1.0);
                } catch (Exception e) {
                    System.out.println("Wrong index: " + j);
                }
            }
            // labels
            // Long label = this.labels.get(indexStart + i);
            BugzillaResolveTime label = this.labelClasses.get(indexStart + i);
            labels.putScalar(i, label.ordinal(), 1.0);

            // mask
            sequenceInputMask.put(
                    new INDArrayIndex[]{
                            NDArrayIndex.point(i),
                            NDArrayIndex.interval(0, len)
                    },
                    Nd4j.ones(len));

        }
        return new org.nd4j.linalg.dataset.DataSet(
                sequenceInput, labels,
                sequenceInputMask, null);
    }

    @Override
    public int inputColumns() {
        return getNumSequenceFeatures();
    }

    @Override
    public int totalOutcomes() {
        return numLabelClasses;
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

    public int getCharSequenceMaxLength() {
        return charSequenceMaxLength;
    }

}
