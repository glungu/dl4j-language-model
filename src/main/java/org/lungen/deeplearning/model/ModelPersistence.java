package org.lungen.deeplearning.model;

import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;

import java.io.IOException;
import java.text.DecimalFormat;
import java.text.SimpleDateFormat;
import java.util.Date;

/**
 * Created by user on 10.11.2018.
 */
public class ModelPersistence {

    public static final SimpleDateFormat DATE_FORMAT = new SimpleDateFormat("YYYYMMdd-HHmmss");
    public static final DecimalFormat NUMBER_FORMAT = new DecimalFormat("#.00");

    public static void save(Model model) {
        try {
            System.out.println("\n\nSaving Model");
            String currentDir = System.getProperty("user.dir");
            String date = DATE_FORMAT.format(new Date());
            String score = NUMBER_FORMAT.format(model.score());

            String fileToSave = currentDir + "/src/main/resources/model-" + date
                    + "-score-" + score + ".zip";

            //Where to save the network. Note: the file is in .zip format - can be opened externally
            ModelSerializer.writeModel(model, fileToSave, true);
            System.out.println("Model saved: " + fileToSave);
        } catch (IOException e) {
            System.out.println("Cannot save model");
            e.printStackTrace();
        }
    }

    public static ComputationGraph loadGraph(String fileName) {
        try {
            long startNano = System.nanoTime();

            System.out.println("\n\nLoading Model");
            String currentDir = System.getProperty("user.dir");
            String fileLocation = currentDir + "/src/main/resources/" + fileName;
            System.out.println("### " + fileLocation);

            // Load the model
            ComputationGraph graph = ModelSerializer.restoreComputationGraph(fileLocation);

            System.out.println("Model loaded in: " + ((System.nanoTime() - startNano)/1e+9) + " seconds");
            return graph;
        } catch (IOException e) {
            System.out.println("Cannot restore model");
            e.printStackTrace();
            throw new IllegalStateException();
        }
    }

    public static MultiLayerNetwork loadNet(String fileName) {
        try {
            System.out.println("\n\nLoading Model");
            String currentDir = System.getProperty("user.dir");
            String fileLocation = currentDir + "/src/main/resources/" + fileName;
            System.out.println("### " + fileLocation);

            // Load the model
            return ModelSerializer.restoreMultiLayerNetwork(fileLocation);
        } catch (IOException e) {
            System.out.println("Cannot restore model");
            e.printStackTrace();
            throw new IllegalStateException();
        }
    }
}
