package org.lungen.deeplearning.model;

import java.io.File;
import java.io.IOException;
import java.text.DecimalFormat;
import java.text.DecimalFormatSymbols;
import java.text.SimpleDateFormat;
import java.util.Date;

import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * ModelPersistence.
 * Utility class for saving/loading model from file.
 *
 */
public class ModelPersistence {

    private static Logger log = LoggerFactory.getLogger("model");
    private static final double NANOS_IN_SECOND = 1e+9;

    private static final SimpleDateFormat DATE_FORMAT = new SimpleDateFormat("YYYYMMdd-HHmmss");
    private static final DecimalFormat NUMBER_FORMAT = new DecimalFormat("#.00");

    private ModelPersistence() {
    }

    static {
        DecimalFormatSymbols symbols = new DecimalFormatSymbols();
        symbols.setDecimalSeparator('.');
        NUMBER_FORMAT.setDecimalFormatSymbols(symbols);
    }

    private static File getCurrentDir() {
        return new File(System.getProperty("user.dir"));
    }

    public static void save(String name, Model model) {
        log.info("Saving Model...");
        String date = DATE_FORMAT.format(new Date());
        String score = NUMBER_FORMAT.format(model.score());

        String modelName = name != null ? name : "unknown";

        String fileName = "net-" + modelName + "-" + date + "-score-" + score + ".zip";
        File file = new File(getCurrentDir(), fileName);

        try {
            ModelSerializer.writeModel(model, file, true);
            log.info("Model saved: " + file);
        } catch (Exception e) {
            throw new IllegalStateException("Cannot save model: " + file.getAbsolutePath(), e);
        }
    }

    public static ComputationGraph loadGraph(String fileName) {
        long startNano = System.nanoTime();
        File file = new File(getCurrentDir(), fileName);
        log.info("Loading Model... " + file.getAbsolutePath());

        try {
            ComputationGraph graph = ModelSerializer.restoreComputationGraph(file);
            log.info("Model loaded in: " + ((System.nanoTime() - startNano)/NANOS_IN_SECOND) + " seconds");
            return graph;
        } catch (IOException e) {
            throw new IllegalStateException("Cannot restore model: " + file.getAbsolutePath(), e);
        }
    }

    public static MultiLayerNetwork loadNet(String fileName) {
        long startNano = System.nanoTime();
        File file = new File(getCurrentDir(), fileName);
        log.info("Loading Model... " + file.getAbsolutePath());

        try {
            MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(file);
            log.info("Model loaded in: " + ((System.nanoTime() - startNano)/NANOS_IN_SECOND) + " seconds");
            return model;
        } catch (IOException e) {
            throw new IllegalStateException("Cannot restore model: " + file.getAbsolutePath(), e);
        }
    }
}
