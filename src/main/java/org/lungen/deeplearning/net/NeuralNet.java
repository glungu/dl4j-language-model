package org.lungen.deeplearning.net;

import org.lungen.deeplearning.iterator.AutoEncoderCharacterIterator;

import java.util.Map;

/**
 * Created by user on 17.11.2018.
 */
public interface NeuralNet {

    String PARAM_MODEL_NAME                     = "MODEL NAME";

    String PARAM_DATA_FILE                      = "DATA FILE";
    String PARAM_MINIBATCH_SIZE                 = "MINIBATCH SIZE";
    String PARAM_SEQUENCE_LENGTH                = "SEQUENCE LENGTH";
    String PARAM_NUMBER_INPUT_FEATURES          = "NUMBER INPUT FEATURES";

    String PARAM_LEARNING_RATE                  = "LEARNING RATE";
    String PARAM_L2_REGULARIZATION              = "L2 REGULARIZATION PARAMETER";
    String PARAM_TRUNCATED_BPTT_SIZE            = "TRUNCATED BPTT SIZE";
    String PARAM_NUMBER_ITER_NO_IMPROVE_STOP    = "ITERATIONS NO IMPROVEMENT STOP";
    String PARAM_MIN_EPOCHS_STOP                = "MIN EPOCHS STOP";

    String PARAM_NUMBER_EPOCHS                  = "NUMBER EPOCHS";
    String PARAM_CHECK_EACH_NUMBER_MINIBATCHES  = "CHECK EACH NUMBER MINIBATCHES";
    String PARAM_STOP_AFTER_NUMBER_MINIBATCHES  = "STOP AFTER NUMBER MINIBATCHES";
    String PARAM_TEMPERATURE                    = "TEMPERATURE";

    Map<String, Object> defaultParams();

    Object iterator(Map<String, Object> params);

    void init(Map<String, Object> params);

    void train(Map<String, Object> params);

    double getBestScore();
}
