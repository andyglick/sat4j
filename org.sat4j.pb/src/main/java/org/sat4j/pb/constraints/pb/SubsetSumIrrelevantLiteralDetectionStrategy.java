package org.sat4j.pb.constraints.pb;

import java.math.BigInteger;

import org.sat4j.specs.IVec;
import org.sat4j.specs.IVecInt;

public class SubsetSumIrrelevantLiteralDetectionStrategy
        implements IrrelevantLiteralDetectionStrategy {

    private final SubsetSum subsetSum = new SubsetSum(20005, 1005);

    @Override
    public boolean dependsOn(int nVars, IVecInt literals,
            IVec<BigInteger> coefficients, BigInteger degree, int literalIndex,
            BigInteger coefficient) {

        int[] elts = new int[literals.size() - 1];
        for (int i = 0, index = 0; i < literals.size(); i++) {
            if (i != literalIndex) {
                elts[index] = coefficients.get(i).intValue();
                index++;
            }
        }

        int intDegree = degree.intValue();
        int minSum = intDegree - coefficient.intValue();
        subsetSum.setElements(elts);

        for (int i = intDegree - 1; i >= minSum; i--) {
            if (subsetSum.sumExists(i)) {
                return true;
            }
        }

        return false;
    }

}
