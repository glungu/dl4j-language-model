package org.lungen.deeplearning.net.predictor;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.lungen.deeplearning.model.ModelPersistence;

/**
 * CharacterSequenceValuePredictor
 *
 * @author lungen.tech@gmail.com
 */
public class CharacterSequenceValuePredictor {

    public static void main(String[] args) {
        String path = "C:/DATA/Projects/dl4j-language-model/net-wiktionary-20190104-112814-score-.82.zip";
        ComputationGraph net = ModelPersistence.loadGraph(path);
    }
}
