package org.lungen.deeplearning.listener;

import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.optimize.api.BaseTrainingListener;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.Serializable;

/**
 * ScorePrintListener
 * Training listener that logs current model score.
 *
 */
public class ScorePrintListener extends BaseTrainingListener implements Serializable {

    private static final Logger log = LoggerFactory.getLogger("listener.score");

    private int printIterations = 10;
    private int epoch;

    /**
     * @param printIterations    frequency with which to print scores (i.e., every printIterations parameter updates)
     */
    public ScorePrintListener(int printIterations) {
        this.printIterations = printIterations;
    }

    /** Default constructor printing every 10 iterations */
    public ScorePrintListener() {}

    public void setEpoch(int epoch) {
        this.epoch = epoch;
    }

    @Override
    public void iterationDone(Model model, int iteration, int epoch) {
        if (printIterations <= 0) {
            printIterations = 1;
        }

        if (iteration % printIterations == 0) {
            double score = model.score();
            log.info("[{}] Score at iteration {} is {}", this.epoch, iteration, score);
        }
    }
}
