package org.lungen.deeplearning.net;

import org.lungen.deeplearning.iterator.AutoEncoderCharacterIterator;

import java.util.Map;

/**
 * Created by user on 17.11.2018.
 */
public interface NeuralNet {

    String PARAM_MODEL_NAME                     = "model.name";

    String PARAM_DATA_FILE                      = "data.file";
    String PARAM_SEQUENCE_LENGTH                = "data.input.sequence.length";
    String PARAM_NUMBER_INPUT_FEATURES          = "data.input.features";
    String PARAM_NUMBER_OUTPUT_CLASSES          = "data.output.classes";

    String PARAM_NUMBER_EPOCHS                  = "training.epochs";
    String PARAM_MINIBATCH_SIZE                 = "training.minibatch.size";
    String PARAM_LEARNING_RATE                  = "training.learning.rate";
    String PARAM_L2_REGULARIZATION              = "training.regularization.l2";
    String PARAM_TRUNCATED_BPTT_SIZE            = "training.backprop.tbptt.size";

    String PARAM_CHECK_EACH_NUMBER_MINIBATCHES  = "training.evaluate.minibatches";
    String PARAM_TEMPERATURE                    = "training.evaluate.temperature";

    String PARAM_NUMBER_ITER_NO_IMPROVE_STOP    = "training.stop.no_improvements.iterations";
    String PARAM_MIN_EPOCHS_STOP                = "training.stop.min_epochs";
    String PARAM_STOP_AFTER_NUMBER_MINIBATCHES  = "training.stop.minibatches";



    Map<String, Object> defaultParams();

    Object iterator(Map<String, Object> params);

    void init(Map<String, Object> params);

    void train(Map<String, Object> params);

    double getBestScore();
}
