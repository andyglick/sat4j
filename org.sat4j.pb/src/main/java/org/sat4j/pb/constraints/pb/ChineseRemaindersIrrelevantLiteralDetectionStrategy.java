package org.sat4j.pb.constraints.pb;

import java.math.BigInteger;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.sat4j.specs.IVec;
import org.sat4j.specs.IVecInt;

/**
 * 
 * @author Romain WALLON
 */
public class ChineseRemaindersIrrelevantLiteralDetectionStrategy
        implements IrrelevantLiteralDetectionStrategy {

    private final List<IrrelevantLiteralDetectionStrategy> strategies;

    public ChineseRemaindersIrrelevantLiteralDetectionStrategy(
            IrrelevantLiteralDetectionStrategy... strategies) {
        this(Arrays.asList(strategies));
    }

    public ChineseRemaindersIrrelevantLiteralDetectionStrategy(
            List<IrrelevantLiteralDetectionStrategy> strategies) {
        this.strategies = strategies;
    }

    public static ChineseRemaindersIrrelevantLiteralDetectionStrategy forPrimes(
            int... primes) {
        List<IrrelevantLiteralDetectionStrategy> strategies = new ArrayList<>(
                primes.length);
        for (int p : primes) {
            strategies.add(
                    new SubsetSumApproximationIrrelevantLiteralDetectionStrategy(
                            p));
        }
        return new ChineseRemaindersIrrelevantLiteralDetectionStrategy(
                strategies);
    }

    @Override
    public boolean dependsOn(int nVars, IVecInt literals,
            IVec<BigInteger> coefficients, BigInteger degree, int literalIndex,
            BigInteger coefficient) {
        for (IrrelevantLiteralDetectionStrategy strategy : strategies) {
            if (!strategy.dependsOn(nVars, literals, coefficients, degree,
                    literalIndex, coefficient)) {
                // In this case, we are sure that the literal is irrelevant.
                return false;
            }
        }

        return true;
    }

}
