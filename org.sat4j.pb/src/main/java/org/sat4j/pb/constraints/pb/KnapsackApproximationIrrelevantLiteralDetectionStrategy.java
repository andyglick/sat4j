package org.sat4j.pb.constraints.pb;

import java.math.BigInteger;

import org.sat4j.specs.IVec;
import org.sat4j.specs.IVecInt;

public class KnapsackApproximationIrrelevantLiteralDetectionStrategy
        implements IrrelevantLiteralDetectionStrategy {

    private static final int MAX_VALUE = 547;

    private static final BigInteger MAX_VALUE_BIGINT = BigInteger
            .valueOf(MAX_VALUE);

    private static final BigInteger EPSILON_INV = BigInteger.valueOf(100);

    private final Knapsack knapsack = new Knapsack(MAX_VALUE + 5,
            IrrelevantLiteralRemover.MAX_LITERALS + 5);

    @Override
    public boolean dependsOn(int nVars, IVecInt literals,
            IVec<BigInteger> coefficients, BigInteger degree, int literalIndex,
            BigInteger coefficient) {
        // Looking for the maximum value.
        int size = 0;
        BigInteger max = BigInteger.ZERO;
        for (int i = 0; i < literals.size(); i++) {
            if (i != literalIndex && !coefficients.get(i).equals(degree)) {
                max = max.max(coefficients.get(i));
                size++;
            }
        }
        if (size == 0) {
            return coefficients.get(literalIndex).compareTo(degree) >= 0;
        }
        // Reducing the sizes.
        BigInteger n = BigInteger.valueOf(size);
        int[] values = new int[size];
        BigInteger[] weights = new BigInteger[size];
        for (int i = 0, index = 0; i < literals.size(); i++) {
            if (i != literalIndex && !coefficients.get(i).equals(degree)) {
                BigInteger vHat = EPSILON_INV.multiply(n)
                        .multiply(coefficients.get(i)).divide(max);
                if (vHat.compareTo(MAX_VALUE_BIGINT) > 0) {
                    // Value too big to be treated.
                    return true;
                }
                values[index] = vHat.intValue();
                weights[index] = coefficients.get(i);
                index++;
            }
        }
        knapsack.setValues(values, weights);
        BigInteger best = BigInteger
                .valueOf(knapsack.bestValue(degree.subtract(BigInteger.ONE)));
        best = best.multiply(max.divide(EPSILON_INV.multiply(n)));
        return best.compareTo(degree.subtract(coefficient)) >= 0;
    }

}
