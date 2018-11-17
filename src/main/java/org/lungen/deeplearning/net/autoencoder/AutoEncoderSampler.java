package org.lungen.deeplearning.net.autoencoder;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.graph.vertex.GraphVertex;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.lungen.deeplearning.iterator.AutoEncoderCharacterIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.Random;

public class AutoEncoderSampler {

    private ComputationGraph net;
    private AutoEncoderCharacterIterator iterator;

    public AutoEncoderSampler(ComputationGraph net, AutoEncoderCharacterIterator iterator) {
        this.net = net;
        this.iterator = iterator;
    }

    public void output(MultiDataSet batchDS) {
        INDArray inputFeatures = batchDS.getFeatures()[0];
        String inputString = iterator.recordToString(inputFeatures);
        System.out.println("--- Original ---");
        System.out.println(inputString);
        System.out.println("----------------");

        // sample starts here
        net.rnnClearPreviousState();

//        INDArray in = Nd4j.create(ArrayUtils.toPrimitive(rowIn.toArray(new Double[0])), new int[] { 1, 1, rowIn.size() });
        int dictSize = (int) inputFeatures.size(1);
        int sequenceSize = (int) inputFeatures.size(2);
        INDArray in = inputFeatures.get(NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.all())
                .reshape(1, dictSize, sequenceSize);
        Random rnd = new Random(7);

//        double[] decodeArr = new double[dictSize];
//        decodeArr[2] = 1;
//        INDArray decode = Nd4j.create(decodeArr, new int[] { 1, dictSize, 1 });
        INDArray decode = inputFeatures.get(NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.point(0))
                .reshape(1, dictSize, 1);

        net.feedForward(new INDArray[] { in, decode }, false, false);

        org.deeplearning4j.nn.layers.recurrent.LSTM decoder =
                (org.deeplearning4j.nn.layers.recurrent.LSTM) net.getLayer("decoder");

        Layer output = net.getLayer("output");
        GraphVertex mergeVertex = net.getVertex("merge");
        INDArray thoughtVector = mergeVertex.getInputs()[1];
        LayerWorkspaceMgr mgr = LayerWorkspaceMgr.noWorkspaces();

        for (int row = 0; row < sequenceSize; ++row) {
            mergeVertex.setInputs(decode, thoughtVector);
            INDArray merged = mergeVertex.doForward(false, mgr);
            INDArray activateDec = decoder.rnnTimeStep(merged, mgr);
            INDArray out = output.activate(activateDec, false, mgr);
            double d = rnd.nextDouble();
            double sum = 0.0;
            int idx = -1;
            for (int s = 0; s < out.size(1); s++) {
                sum += out.getDouble(0, s, 0);
                if (d <= sum) {
                    idx = s;
                    char c = iterator.convertIndexToCharacter(idx);
                    System.out.print(c);
//                    if (printUnknowns || s != 0) {
//                        System.out.print(revDict.get((double) s) + " ");
//                    }
                    break;
                }
            }
//            if (idx == 1) {
//                break;
//            }
            double[] newDecodeArr = new double[dictSize];
            newDecodeArr[idx] = 1;
            decode = Nd4j.create(newDecodeArr, new int[] { 1, dictSize, 1 });
        }
        System.out.println();
    }
}
