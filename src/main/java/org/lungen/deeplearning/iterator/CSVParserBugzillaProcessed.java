package org.lungen.deeplearning.iterator;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.stream.IntStream;

import org.json.JSONArray;
import org.json.JSONException;

/**
 * CSVParserBugzillaProcessed
 *
 * @author lungen.tech@gmail.com
 */
public class CSVParserBugzillaProcessed extends CSVParser {

    private static final SimpleDateFormat DF = new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss'Z'");
    private final File workDir;

    public CSVParserBugzillaProcessed(File csvFile, boolean hasHeaders) {
        super(csvFile, hasHeaders);
        this.workDir = csvFile.getParentFile();
    }

    public void process() {
        String[] columnNames = {"status", "resolution", "severity", "priority", "creator", "assigned_to", "component"};
        Map<String, Map<String, Integer>> categoricalMap = categoricalStringToNumeric(columnNames);

        // process lines
        parsedLines.forEach(parsedLine -> {
            categoricalStringToNumeric(parsedLine, categoricalMap, columnNames);
            dateToNumeric(parsedLine, "creation_time", "resolution_time");
            arrayToFirstValue(parsedLine, "blocks", "depends_on");
        });

        // categorical values mapping to dictionary files
        categoricalMap.keySet().forEach(name -> {
            Map<String, Integer> valuesMap = categoricalMap.get(name);
            File dictFile = new File(workDir, "dict-" + name + ".txt");
            int MAXLEN = valuesMap.keySet().stream().mapToInt(String::length).max().orElse(0) + 1;

            try (FileOutputStream dictFileStream = new FileOutputStream(dictFile)) {
                valuesMap.entrySet().stream().sorted(Map.Entry.comparingByValue()).forEach(entry -> {
                    String key = entry.getKey();
                    try {
                        StringBuilder indent = new StringBuilder();
                        IntStream.range(0, MAXLEN - key.length()).forEach(n -> indent.append(" "));
                        dictFileStream.write((key + ": " + indent.toString() + entry.getValue() + "\n").getBytes());
                    } catch (IOException e) {
                        log.error("Cannot write to file: " + dictFile, e);
                    }
                });
            } catch (Exception e) {
                log.error("Cannot open file: " + dictFile, e);
            }
        });
    }

    private void arrayToFirstValue(List<String> parsedLine, String... columnNames) {
        Arrays.stream(columnNames).forEach(columnName -> {
            Integer index = headers.get(columnName);
            String value = parsedLine.get(index);
            try {
                JSONArray array = new JSONArray(unQuote(value));
                String newValue = array.isEmpty() ? "" : array.get(0) + "";
                parsedLine.set(index, newValue);
            } catch (JSONException e) {
                log.error("Error parsing JSON", e);
            }
        });
    }

    private void categoricalStringToNumeric(
            List<String> parsedLine, Map<String, Map<String, Integer>> categoricalMap,
            String... columnNames) {

        Arrays.stream(columnNames).forEach(columnName -> {
            Integer index = headers.get(columnName);
            String value = parsedLine.get(index);
            Integer numericalValue = categoricalMap.get(columnName).get(value);
            parsedLine.set(index, numericalValue + "");
        });
    }

    private void dateToNumeric(List<String> parsedLine, String... columnNames) {
        Arrays.stream(columnNames).forEach(columnName -> {
            Integer ind = headers.get(columnName);
            if (parsedLine.size() > ind) {
                String dateString = parsedLine.get(ind);
                try {
                    String numericString = "";
                    if (!dateString.equals("[]")) {
                        numericString = DF.parse(dateString).getTime() + "";
                    }
                    parsedLine.set(ind, numericString);
                } catch (Exception e) {
                    log.error("Cannot parse date, id: " + parsedLine.get(headers.get("id")) + ", "
                            + "index: " + ind + ", value: " + dateString);
                }
            } else {
                log.error("Cannot parse date, id: " + parsedLine.get(headers.get("id")) + ", "
                        + "no such index: " + ind + ", size: " + parsedLine.size());
            }
        });
    }

    public static void main(String[] args) {
        String dir = "C:/Work/ML/bugzilla";
        File sourceFile = new File(dir, "bugzilla-processed.csv");

        CSVParserBugzillaProcessed parser = new CSVParserBugzillaProcessed(sourceFile, true);
        parser.process();

        File processedFile = new File(dir, "bugzilla-processed-final.csv");
        parser.save(processedFile);
    }
}
