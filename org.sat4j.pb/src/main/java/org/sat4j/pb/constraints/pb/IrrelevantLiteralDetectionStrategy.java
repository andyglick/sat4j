package org.sat4j.pb.constraints.pb;

import java.math.BigInteger;

import org.sat4j.specs.IVecInt;

public interface IrrelevantLiteralDetectionStrategy {

    boolean dependsOn(int nVars, IVecInt literals, BigInteger[] coefficients,
            BigInteger degree, int literalIndex, BigInteger coefficient);

}
