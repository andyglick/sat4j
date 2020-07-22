package org.sat4j.pb.constraints.pb;

import java.math.BigInteger;

import org.sat4j.core.Vec;
import org.sat4j.core.VecInt;
import org.sat4j.pb.IPBSolver;
import org.sat4j.pb.SolverFactory;
import org.sat4j.specs.ContradictionException;
import org.sat4j.specs.IVec;
import org.sat4j.specs.IVecInt;
import org.sat4j.specs.TimeoutException;

public class SatBasedIrrelevantLiteralDetectionStrategy
        implements IrrelevantLiteralDetectionStrategy {

    private final IPBSolver solver;

    public SatBasedIrrelevantLiteralDetectionStrategy(IPBSolver solver) {
        this.solver = solver;
    }

    public static IrrelevantLiteralDetectionStrategy newCuttingPlanes() {
        return new SatBasedIrrelevantLiteralDetectionStrategy(
                SolverFactory.newCuttingPlanes());
    }

    public static IrrelevantLiteralDetectionStrategy newResolution() {
        return new SatBasedIrrelevantLiteralDetectionStrategy(
                SolverFactory.newResolution());
    }

    @Override
    public boolean dependsOn(int nVars, IVecInt literals,
            BigInteger[] coefficients, BigInteger degree, int literalIndex,
            BigInteger coefficient) {
        try {
            solver.reset();
            solver.setTimeout(5);
            solver.newVar(nVars);

            IVecInt lits = new VecInt();
            IVec<BigInteger> coeffs = new Vec<>();
            for (int i = 0; i < literals.size(); i++) {
                if (i != literalIndex) {
                    lits.push(literals.get(i));
                    coeffs.push(coefficients[i]);
                }

            }
            solver.addAtMost(lits, coeffs, degree.subtract(BigInteger.ONE));
            solver.addAtLeast(lits, coeffs, degree.subtract(coefficient));
            return solver.isSatisfiable();

        } catch (ContradictionException e) {
            return false;

        } catch (TimeoutException e) {
            return true;
        }
    }

}
