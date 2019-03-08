import java.math.BigInteger;

import org.sat4j.specs.IVec;
import org.sat4j.specs.IVecInt;

public interface IrrelevantLiteralDetectionStrategy {

    boolean dependsOn(IVecInt literals, IVec<BigInteger> coefficients,
            BigInteger degree, int literalIndex, BigInteger coefficient);

}
