package org.sat4j.pb.constraints.pb;

import java.math.BigInteger;

import org.sat4j.pb.core.PBSolverStats;

public class NoIrrelevantLiteralRemover implements IIrrelevantLiteralRemover {

    private static final IIrrelevantLiteralRemover INSTANCE = new NoIrrelevantLiteralRemover();

    private NoIrrelevantLiteralRemover() {
        // TODO Auto-generated constructor stub
    }

    public static IIrrelevantLiteralRemover instance() {
        return INSTANCE;
    }

    @Override
    public void postProcess(int dl, ConflictMap conflictMap) {
        // TODO Auto-generated method stub

    }

    @Override
    public BigInteger remove(int nVars, int[] literals,
            BigInteger[] coefficients, BigInteger degree, PBSolverStats stats,
            boolean conflict) {
        // TODO Auto-generated method stub
        return degree;
    }

}
