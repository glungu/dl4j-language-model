package org.lungen.deeplearning.net;

import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;

import java.util.Map;

/**
 * Created by user on 17.11.2018.
 */
public interface NeuralNet {

    public static final String PARAM_LEARNING_RATE                  = "LEARNING RATE";
    public static final String PARAM_L2_REGULARIZATION              = "L2 REGULARIZATION PARAMETER";
    public static final String PARAM_TRUNCATED_BPTT_SIZE            = "TRUNCATED BPTT SIZE";
    public static final String PARAM_NUMBER_INPUT_FEATURES          = "NUMBER INPUT FEATURES";

    public static final String PARAM_ITERATOR                       = "ITERATOR";
    public static final String PARAM_NUMBER_EPOCHS                  = "NUMBER EPOCHS";
    public static final String PARAM_CHECK_EACH_NUMBER_MINIBATCHES  = "CHECK EACH NUMBER MINIBATCHES";
    public static final String PARAM_STOP_AFTER_NUMBER_MINIBATCHES  = "STOP AFTER NUMBER MINIBATCHES";

    public void init(Map<String, Object> initParams);

    public void train(Map<String, Object> trainParams);

}
