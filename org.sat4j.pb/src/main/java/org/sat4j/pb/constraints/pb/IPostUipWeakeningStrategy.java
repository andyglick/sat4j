package org.sat4j.pb.constraints.pb;

import java.math.BigInteger;

import org.sat4j.minisat.core.ILits;

@FunctionalInterface
public interface IPostUipWeakeningStrategy {

    int findLiteralToRemove(ILits voc, PBConstr wpb, BigInteger[] coefsBis,
            int indLitImplied, BigInteger degreeBis, BigInteger slack,
            BigInteger coeffAssertive, int assertiveDL);

}
