package org.sat4j.pb.constraints.pb;

import java.math.BigInteger;
import java.util.Arrays;

import org.sat4j.specs.IVec;
import org.sat4j.specs.IVecInt;

public class SubsetSumApproximationIrrelevantLiteralDetectionStrategy
        implements IrrelevantLiteralDetectionStrategy {

    private static final int DEFAULT_MAX_DEGREE = 4547;

    private final int maxDegree;

    private final BigInteger maxDegreeAsBigInteger;

    private final SubsetSumModulo subsetSum;

    public SubsetSumApproximationIrrelevantLiteralDetectionStrategy() {
        this(DEFAULT_MAX_DEGREE);
    }

    public SubsetSumApproximationIrrelevantLiteralDetectionStrategy(
            int maxDegree) {
        this.maxDegree = maxDegree;
        this.maxDegreeAsBigInteger = BigInteger.valueOf(maxDegree);
        this.subsetSum = new SubsetSumModulo(maxDegree);
    }

    @Override
    public boolean dependsOn(int nVars, IVecInt literals,
            IVec<BigInteger> coefficients, BigInteger degree, int literalIndex,
            BigInteger coefficient) {
        int[] elts = new int[literals.size() - 1];
        int index = 0;
        for (int i = 0; i < literals.size(); i++) {
            if (i != literalIndex
                    && coefficients.get(i).compareTo(degree) < 0) {
                elts[index] = mod(coefficients.get(i).min(degree));
                index++;
            }
        }
        if (index == 0) {
            return coefficient.compareTo(degree) >= 0;
        }
        subsetSum.setElements(Arrays.copyOf(elts, index));
        BigInteger minSum = degree.subtract(coefficient.min(degree));

        for (BigInteger i = degree.subtract(BigInteger.ONE); i
                .compareTo(minSum) >= 0; i = i.subtract(BigInteger.ONE)) {
            if (subsetSum.sumExists(mod(i))) {
                return true;
            }
        }

        return false;
    }

    private int mod(BigInteger b) {
        if (b.bitLength() < Long.SIZE) {
            // Safe cast since the result is necessarily less than the modulo.
            return (int) (b.longValue() % maxDegree);
        }

        // Safe conversion since the result is necessarily less than the modulo.
        return b.mod(maxDegreeAsBigInteger).intValue();
    }

}
