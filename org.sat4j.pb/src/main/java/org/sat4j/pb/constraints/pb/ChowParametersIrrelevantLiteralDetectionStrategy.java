package org.sat4j.pb.constraints.pb;

import java.math.BigInteger;

import org.sat4j.specs.IVec;
import org.sat4j.specs.IVecInt;

public class ChowParametersIrrelevantLiteralDetectionStrategy
        implements IrrelevantLiteralDetectionStrategy {

    private final ChowParameters chowParameters = new ChowParameters(1005,
            20005);

    @Override
    public boolean dependsOn(int nVars, IVecInt literals,
            IVec<BigInteger> coefficients, BigInteger degree, int literalIndex,
            BigInteger coefficient) {
        chowParameters.setConstraint(literals, coefficients, degree);
        return chowParameters.dependsOn(literalIndex);
    }

}
