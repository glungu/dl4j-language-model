package org.lungen.deeplearning.listener;

import java.io.IOException;

import org.deeplearning4j.earlystopping.saver.InMemoryModelSaver;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.lungen.deeplearning.model.ModelPersistence;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class EarlyStopListener extends ScoreIterationListener {

    private static final Logger log = LoggerFactory.getLogger("listener.earlystop");

    private InMemoryModelSaver<Model> modelSaver = new InMemoryModelSaver<>();

    private double bestScore = Double.MAX_VALUE;

    private int iterationsWithoutImprovement;

    private int iterationsWithoutImprovementLimit;

    private boolean stopRecommended;

    public EarlyStopListener(int iterationsWithoutImprovementLimit) {
        this.iterationsWithoutImprovementLimit = iterationsWithoutImprovementLimit;
    }

    @Override
    public void iterationDone(Model model, int iteration, int epoch) {
        try {

            double score = model.score();
            if (bestScore > score) {
                bestScore = score;
                iterationsWithoutImprovement = 0;
                modelSaver.saveBestModel(model, score);
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

    public boolean isStopRecommended() {
        return stopRecommended;
    }

    public void writeBestModel() {
        try {
            Model bestModel = modelSaver.getBestModel();
            ModelPersistence.save(bestModel);

        } catch (IOException e) {
            log.error("Cannot save model", e);
        }
    }
}
