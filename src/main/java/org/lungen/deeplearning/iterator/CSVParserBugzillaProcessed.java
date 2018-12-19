package org.lungen.deeplearning.iterator;

import org.json.JSONArray;
import org.json.JSONException;

import java.io.File;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

/**
 * CSVParserBugzillaProcessed
 *
 * @author lungen.tech@gmail.com
 */
public class CSVParserBugzillaProcessed extends CSVParser {

    private static final SimpleDateFormat DF = new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss'Z'");

    public CSVParserBugzillaProcessed(File csvFile, boolean hasHeaders) {
        super(csvFile, hasHeaders);
    }

    public void process() {
        String[] columnNames = {"status", "resolution", "severity", "priority", "creator", "assigned_to", "component"};
        Map<String, Map<String, Integer>> categoricalMap = categoricalStringToNumeric(columnNames);
        // output categorical values mapping
        categoricalMap.keySet().forEach(name -> {
            Map<String, Integer> valuesMap = categoricalMap.get(name);
            System.out.println("Categorical feature '" + name + "': ");
            valuesMap.keySet().forEach(i -> System.out.println(i + " -> " + valuesMap.get(i)));
        });
        parsedLines.forEach(parsedLine -> {
            categoricalStringToNumeric(parsedLine, categoricalMap, columnNames);
            dateToNumeric(parsedLine, "creation_time", "resolution_time");
            arrayToFirstValue(parsedLine, "blocks", "depends_on");
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

                if (array.length() > 1) {
                    System.out.println("Array more than one element: " + parsedLine.get(headers.get("id")));
                }
            } catch (JSONException e) {
                e.printStackTrace();
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
            try {
                String dateString = parsedLine.get(ind);
                String numericString = "";
                if (!dateString.equals("[]")) {
                    numericString = DF.parse(dateString).getTime() + "";
                }
                parsedLine.set(ind, numericString);
            } catch (Exception e) {
                log.error("Cannot parse date: " + ind);
            }
        });
    }

    public static void main(String[] args) {
        String dir = "C:/DATA/Projects/DataSets/Jnetx_Bugzilla";
        File sourceFile = new File(dir, "bugzilla-jnetx-processed.csv");

        CSVParserBugzillaProcessed parser = new CSVParserBugzillaProcessed(sourceFile, true);
        parser.process();

        File processedFile = new File(dir, "bugzilla-jnetx-processed-final.csv");
        parser.save(processedFile);
    }
}
