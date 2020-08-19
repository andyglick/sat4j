package org.sat4j.pb.constraints.pb;

import java.math.BigInteger;

import org.sat4j.pb.core.PBSolverStats;

public interface IIrrelevantLiteralRemover extends IPostProcess {

    BigInteger remove(int nVars, int[] literals, BigInteger[] coefficients,
            BigInteger degree, PBSolverStats stats, boolean conflict);

}