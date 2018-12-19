package org.lungen.deeplearning.iterator;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.util.*;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static java.util.stream.Collectors.counting;

/**
 * CSVParser
 *
 * @author lungen.tech@gmail.com
 */
public class CSVParser {

    protected static final Logger log = LoggerFactory.getLogger("parser.csv");

    protected List<List<String>> parsedLines;
    protected Map<String, Integer> headers;
    protected String[] headerNames;

    public CSVParser(File csvFile, boolean hasHeaders) {
        List<String> lines = preProcess(csvFile, hasHeaders);

        int totalSize = 0;
        parsedLines = new ArrayList<>(lines.size());
        for (String line : lines) {
            totalSize += line.length() + 1;
            parsedLines.add(parseCSVLine(line));
        }

        if (hasHeaders) {
            headers = new HashMap<>();
            List<String> headerLine = parsedLines.remove(0);
            headerNames = new String[headerLine.size()];
            for (int i = 0; i < headerLine.size(); i++) {
                headers.put(headerLine.get(i), i);
                headerNames[i] = headerLine.get(i);
            }
        } else {
            headers = null;
            headerNames = null;
        }


        log.info("File read. Lines: " + lines.size() + ", characters: " + totalSize);
    }

    public List<String> preProcess(File csvFile, boolean hasHeaders) {
        List<String> lines = null;
        try {
            lines = Files.readAllLines(csvFile.toPath(), Charset.forName("utf-8"));
        } catch (IOException e) {
            log.error("Error reading file: " + csvFile, e);
            throw new IllegalStateException("Cannot read from file");
        }
        return lines;
    }

    public List<String> getParsedLine(int index) {
        return parsedLines.get(index);
    }

    public List<List<String>> getParsedLines() {
        return parsedLines;
    }

    public Map<String, Integer> getHeaders() {
        return headers;
    }

    public int getHeaderIndex(String name) {
        Integer i = headers.get(name);
        return i != null ? i : -1;
    }

    static List<String> parseCSVLine(String line) {
        char[] chars = line.toCharArray();

        int colStart = 0;
        int index = 0;
        ParseType type = ParseType.REGULAR;

        List<String> values = new ArrayList<>();

        try {
            while(true) {
                boolean stop = false;
                switch (type) {
                    case REGULAR:
                        index = findNextChar(chars, index, ',');
                        if (index == -1) {
                            index = chars.length;
                            stop = true;
                        }
                        values.add(line.substring(colStart, index));
                        index += 1;
                        break;
                    case STRING:
                        index = findNextSequence(chars, index, "\",".toCharArray());
                        // if "", then keep searching
                        while (true) {
                            int i = 0;
                            while (chars[index - i] == '"') {
                                i++;
                            }
                            if (i % 2 == 0) {
                                index = findNextSequence(chars, index + 1, "\",".toCharArray());
                            } else {
                                break;
                            }
                        }
                        if (index == -1) {
                            index = chars.length - 1;
                            stop = true;
                        }
                        values.add(line.substring(colStart, index + 1));
                        index += 2;
                        break;
                    case OBJECT:
                        index = findNextSequence(chars, index, "}\",".toCharArray());
                        if (index == -1) {
                            index = chars.length - 2;
                            stop = true;
                        }
                        values.add(line.substring(colStart, index + 2));
                        index += 3;
                        break;
                }
                if (stop) {
                    break;
                }
                if (chars[index] == '"' && chars[index + 1] == '{') {
                    type = ParseType.OBJECT;
                } else if (chars[index] == '"') {
                    type = ParseType.STRING;
                } else {
                    type = ParseType.REGULAR;
                }
                colStart = index;
            }
        } catch (Exception e) {
            throw new IllegalArgumentException("Cannot parse line: \n" + line, e);
        }
        return values;
    }

    private static int findNextChar(char[] chars, int index, char charToFind) {
        for (int i = index; i < chars.length; i++) {
            if (chars[i] == charToFind) {
                return i;
            }
        }
        return -1;
    }
    private static int findNextSequence(char[] chars, int index, char[] sequenceToFind) {
        for (int i = index; i < chars.length; i++) {
            if (chars[i] == sequenceToFind[0]) {
                boolean equal = true;
                for (int j = 0; j < sequenceToFind.length; j++) {
                    if (chars[i + j] != sequenceToFind[j]) {
                        equal = false;
                        break;
                    }
                }
                if (equal) {
                    return i;
                }
            }
        }
        return -1;
    }

    private static enum ParseType {
        REGULAR,
        STRING,
        OBJECT
    }

    public Map<String, Map<String, Integer>> categoricalStringToNumeric(String... columnNames) {
        // result: columnName -> categorical mapping (value -> index)
        Map<String, Map<String, Integer>> result = new HashMap<>();

        for (String columnName : columnNames) {
            int index = getHeaderIndex(columnName);
            ArrayList<String> values = new ArrayList<>();
            for (List<String> parsedLine : parsedLines) {
                values.add(parsedLine.get(index));
            }

            Map<String, Integer> valueMap = new HashMap<>();
            Map<String, Long> count = values.stream()
                    .collect(Collectors.groupingBy(Function.identity(), counting()));
            Map<String, Long> countSorted = count.entrySet().stream()
                    .sorted(Map.Entry.comparingByValue(Comparator.reverseOrder()))
                    .collect(Collectors.toMap(
                            Map.Entry::getKey,
                            Map.Entry::getValue,
                            (v1, v2) -> {throw new IllegalStateException();},
                            LinkedHashMap::new));

            countSorted.keySet().forEach(name -> System.out.println(name + " -> " + countSorted.get(name)));

            String[] keys = countSorted.keySet().toArray(new String[0]);
            IntStream.range(0, keys.length).forEach(i -> valueMap.put(keys[i], i));
            result.put(columnName, valueMap);
        }
        return result;
    }

    public void save(File saveFile) {
        try (FileWriter f = new FileWriter(saveFile)) {
            if (headerNames != null) {
                // header
                for (int i = 0; i < headerNames.length; i++) {
                    f.write(headerNames[i]);
                    if (i < headerNames.length - 1) {
                        f.write(',');
                    }
                }
                f.write('\n');
            }
            // data
            for (int i = 0; i < parsedLines.size(); i++) {
                List<String> values = parsedLines.get(i);
                for (int j = 0; j < values.size(); j++) {
                    f.write(values.get(j));
                    if (j < values.size() - 1) {
                        f.write(',');
                    }
                }
                f.write('\n');
            }
        } catch (IOException e) {
            log.error("Cannot wtite to file: " + saveFile, e);
        }

        log.info("File saved. Lines: " + parsedLines.size());
    }

    public static String quote(String str) {
        return "\"" + str + "\"";
    }

    public static String unQuote(String descriptionJson) {
        if (descriptionJson.startsWith("\"") && descriptionJson.endsWith("\"")) {
            descriptionJson = descriptionJson.substring(1, descriptionJson.length() - 1);
        }
        return descriptionJson;
    }


}
