package org.sat4j.pb.constraints.pb;

import java.math.BigInteger;
import java.util.Arrays;

import org.sat4j.specs.IVec;
import org.sat4j.specs.IVecInt;

public class SubsetSumApproximationIrrelevantLiteralDetectionStrategy
        implements IrrelevantLiteralDetectionStrategy {

    private static final int MAX_DEGREE = 4547;
    private static final BigInteger MAX_DEGREE_BIGINT = BigInteger
            .valueOf(MAX_DEGREE);

    private final SubsetSumModulo subsetSum = new SubsetSumModulo(
            RemoveIrrelevantPostProcess.MAX_LITERALS + 5, MAX_DEGREE + 5,
            MAX_DEGREE);

    @Override
    public boolean dependsOn(int nVars, IVecInt literals,
            IVec<BigInteger> coefficients, BigInteger degree, int literalIndex,
            BigInteger coefficient) {
        int[] elts = new int[literals.size() - 1];
        int index = 0;
        for (int i = 0; i < literals.size(); i++) {
            if (i != literalIndex && !coefficients.get(i).equals(degree)) {
                elts[index] = mod(coefficients.get(i));
                index++;
            }
        }

        subsetSum.setElements(Arrays.copyOf(elts, index));
        BigInteger minSum = degree.subtract(coefficient);

        for (BigInteger i = degree.subtract(BigInteger.ONE); i
                .compareTo(minSum) >= 0; i = i.subtract(BigInteger.ONE)) {
            if (subsetSum.sumExists(mod(i))) {
                return true;
            }
        }

        return false;
    }

    private static int mod(BigInteger b) {
        if (b.bitLength() < Long.SIZE) {
            // Safe cast since the result is necessarily less than the modulo.
            return (int) (b.longValue() % MAX_DEGREE);
        }

        // Safe conversion since the result is necessarily less than the modulo.
        return b.mod(MAX_DEGREE_BIGINT).intValue();
    }

}
