package org.lungen.deeplearning.iterator;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.File;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Date;
import java.util.List;
import java.util.stream.IntStream;

/**
 * CSVParserBugzilla
 *
 * @author lungen.tech@gmail.com
 */
public class CSVParserBugzilla extends CSVParser {

    public static final SimpleDateFormat DF = new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss'Z'");

    public CSVParserBugzilla(File csvFile, boolean hasHeaders) {
        super(csvFile, hasHeaders);
    }

    @Override
    public List<String> preProcess(File csvFile, boolean hasHeaders) {
        List<String> lines = super.preProcess(csvFile, hasHeaders);
        List<String> processedLines = new ArrayList<>();
        String procLine = "";
        boolean complete = false;
        for (int i = 0; i < lines.size(); i++) {
            String line = lines.get(i);
            if (hasHeaders && i == 0) {
                processedLines.add(line);
                continue;
            }
            procLine = procLine.length() > 0 ? procLine + "\\n" + line : line;
            complete = line.endsWith(",---");
            if (complete) {
                processedLines.add(procLine);
                procLine = "";
            }
        }
        return processedLines;
    }

    public void process() {
        List<String> keepHeaders = Arrays.asList(
                "id",
                "status",
                "resolution",
                "severity",
                "priority",
                "creator",
                "assigned_to",
                "component",
                "blocks",
                "depends_on",
                "dupe_of",
                "description",
                "creation_time",
                "resolution_dates");
        int[] keepIndexes = keepHeaders.stream().mapToInt(headers::get).toArray();

        List<List<String>> keepLines = new ArrayList<>();

        for (int i = 0; i < parsedLines.size(); i++) {
            List<String> parsedLine = parsedLines.get(i);
            List<String> keepValues = new ArrayList<>();
            Arrays.stream(keepIndexes).forEach(ind -> keepValues.add(parsedLine.get(ind)));
            // parsedLines.set(i, keepValues);


            String summary = parsedLine.get(headers.get("summary"));
            String descriptionJson = parsedLine.get(headers.get("description"));

            descriptionJson = descriptionJson.replaceAll("\"\"", "\"");
            descriptionJson = descriptionJson.replaceAll("\\\\x00", "");
            descriptionJson = descriptionJson.replaceAll("\\\\x01", "");
            descriptionJson = descriptionJson.replaceAll("\\\\x02", "");
            descriptionJson = descriptionJson.replaceAll("\\\\x03", "");
            descriptionJson = descriptionJson.replaceAll("\\\\x04", "");
            descriptionJson = descriptionJson.replaceAll("\\\\x05", "");
            descriptionJson = descriptionJson.replaceAll("\\\\x06", "");
            descriptionJson = descriptionJson.replaceAll("\\\\x07", "");
            descriptionJson = descriptionJson.replaceAll("\\\\x08", "");
            descriptionJson = descriptionJson.replaceAll("\\\\x11", "");
            descriptionJson = descriptionJson.replaceAll("\\\\xa0", "");
            descriptionJson = descriptionJson.replaceAll("\\\\x94", "");
            descriptionJson = descriptionJson.replaceAll("\\\\x1b", "");
            descriptionJson = descriptionJson.replaceAll("\\\\x18", "");
            descriptionJson = descriptionJson.replaceAll("\\\\x0f", "");
            descriptionJson = descriptionJson.replaceAll("\\\\x14", "");
            descriptionJson = descriptionJson.replaceAll("\\\\x19", "");
            descriptionJson = descriptionJson.replaceAll("\\\\x1e", "");
            descriptionJson = descriptionJson.replaceAll("\\\\x1d", "");
            descriptionJson = descriptionJson.replaceAll("\\\\x82", "");
            descriptionJson = descriptionJson.replaceAll("\\\\x9f", "");
            descriptionJson = descriptionJson.replaceAll("\\\\x1c", "");
            descriptionJson = descriptionJson.replaceAll("\\\\x7f", "");
            descriptionJson = descriptionJson.replaceAll("\\\\x98", "");
            descriptionJson = descriptionJson.replaceAll("\\\\x0b", "");
            descriptionJson = descriptionJson.replaceAll("\\\\x15", "");
            descriptionJson = descriptionJson.replaceAll("\\\\x80", "");
            descriptionJson = descriptionJson.replaceAll("\\\\x99", "");

            JSONObject jsonObject = null;
            try {
                jsonObject = new JSONObject(unQuote(descriptionJson));
            } catch (JSONException e) {
                throw new IllegalStateException("Cannot parse json");
            }

            String text = summary + "\n" + jsonObject.get("text");
            text = text.replaceAll("\"", "\"\"");
            String description = ("\"" + text + "\"").replaceAll("\n", "\\\\n");

            // set description
            keepValues.set(keepHeaders.indexOf("description"), description);

            // set resolution time & duration
            String status = parsedLine.get(headers.get("status"));
            String resolutionHours = "-1";
            if (status.equals("CLOSED") || status.equals("RESOLVED")) {
                String creationTime = parsedLine.get(headers.get("creation_time")).trim();
                String resolutionDates = parsedLine.get(headers.get("resolution_dates")).trim();

                try {
                    JSONArray resDates = new JSONArray(unQuote(resolutionDates));
                    if (resDates.length() > 0) {
                        String lastResolution = (String) resDates.get(resDates.length() - 1);
                        keepValues.set(keepHeaders.indexOf("resolution_dates"), lastResolution);

                        Date resolutionDate = DF.parse(lastResolution);
                        Date creationDate = DF.parse(creationTime);
                        long diff = resolutionDate.getTime() - creationDate.getTime();
                        resolutionHours = (diff / (3600000L)) + "";
                    }

                    keepValues.add(resolutionHours);
                    keepLines.add(keepValues);

                } catch (ParseException e) {
                    log.error("Cannot parse date", e);
                }
            }
        }

        this.parsedLines = keepLines;

        List<String> newHeaders = new ArrayList<>(keepHeaders);
        newHeaders.set(newHeaders.indexOf("resolution_dates"), "resolution_time");
        newHeaders.add("resolution_duration_hours");
        headerNames = newHeaders.toArray(new String[0]);

        // re-define indexes
        headers.clear();
        IntStream.range(0, newHeaders.size())
                .forEach(i -> headers.put(newHeaders.get(i), i));

        log.info("File processed. Lines: " + parsedLines.size());
    }

    public static void main(String[] args) {
        String dir = "C:/DATA/Projects/DataSets/Bugzilla";
        File sourceFile = new File(dir, "bugzilla.csv");

        CSVParserBugzilla parser = new CSVParserBugzilla(sourceFile, true);
        parser.process();

        File processedFile = new File(dir, "bugzilla-processed.csv");
        parser.save(processedFile);
    }

}
