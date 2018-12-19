package org.lungen.deeplearning.listener;

import java.io.IOException;

import org.deeplearning4j.earlystopping.saver.InMemoryModelSaver;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.lungen.deeplearning.model.ModelPersistence;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class EarlyStopListener extends ScoreIterationListener {

    private static final Logger log = LoggerFactory.getLogger("listener.earlystop");

    private InMemoryModelSaver<Model> modelSaver = new InMemoryModelSaver<>();

    private double bestScore = Double.MAX_VALUE;

    private int iterationsWithoutImprovement;

    private String name;
    private int iterationsWithoutImprovementLimit;
    private int minNumberEpochs;
    private int epoch;

    private boolean stopRecommended;

    public EarlyStopListener(String name, int iterationsWithoutImprovementLimit, int minNumberEpochs) {
        this.name = name;
        this.iterationsWithoutImprovementLimit = iterationsWithoutImprovementLimit;
        this.minNumberEpochs = minNumberEpochs;
    }

    public void setEpoch(int epoch) {
        this.epoch = epoch;
    }

    @Override
    public void iterationDone(Model model, int iteration, int epoch) {
        if (this.epoch < minNumberEpochs) {
            return;
        }

        try {

            double score = model.score();
            if (bestScore > score) {
                modelSaver.saveBestModel(model, score);
                log.info("Improved model: {} -> {}, saved score {}", bestScore, score, modelSaver.getBestModel().score());
                iterationsWithoutImprovement = 0;
                bestScore = score;
            } else {
                iterationsWithoutImprovement++;
                if (iterationsWithoutImprovement > iterationsWithoutImprovementLimit) {
                    stopRecommended = true;
                    log.info("{} iterations without score improvement. Stop recommended.", iterationsWithoutImprovementLimit);
                }
            }

        } catch (IOException e) {
            log.error("Error processing iteration", e);
        }
    }

    public double getBestScore() {
        return bestScore;
    }

    public boolean isStopRecommended() {
        return stopRecommended;
    }

    public void writeBestModel() {
        try {
            Model bestModel = modelSaver.getBestModel();
            ModelPersistence.save(name, bestModel);

        } catch (IOException e) {
            log.error("Cannot save model", e);
        }
    }
}
