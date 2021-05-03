package org.sat4j.pb.constraints.pb;

import java.math.BigInteger;

import org.sat4j.minisat.core.ILits;

public class NeverWeakenAfterUip implements IPostUipWeakeningStrategy {

    @Override
    public int findLiteralToRemove(ILits voc, PBConstr wpb,
            BigInteger[] coefsBis, int indLitImplied, BigInteger degreeBis,
            BigInteger slack, BigInteger coeffAssertive, int assertiveDL) {
        return -1;
    }

}
