package org.sat4j.pb.constraints.pb;

import java.math.BigInteger;

import org.sat4j.specs.IVec;
import org.sat4j.specs.IVecInt;

public interface IrrelevantLiteralDetectionStrategy {

    static IrrelevantLiteralDetectionStrategy defaultStrategy() {
        return new SubsetSumApproximationIrrelevantLiteralDetectionStrategy();
    }

    boolean dependsOn(int nVars, IVecInt literals,
            IVec<BigInteger> coefficients, BigInteger degree, int literalIndex,
            BigInteger coefficient);

}
