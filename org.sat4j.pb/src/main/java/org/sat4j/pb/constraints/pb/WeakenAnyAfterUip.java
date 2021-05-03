package org.sat4j.pb.constraints.pb;

import java.math.BigInteger;

import org.sat4j.minisat.core.ILits;

public class WeakenAnyAfterUip implements IPostUipWeakeningStrategy {

    @Override
    public int findLiteralToRemove(ILits voc, PBConstr wpb,
            BigInteger[] coefsBis, int indLitImplied, BigInteger degreeBis,
            BigInteger slack, BigInteger coeffAssertive, int assertiveDL) {
        // Any literal assigned at a level > assertiveDL may be removed
        int lit = -1;
        int size = wpb.size();
        for (int ind = 0; (ind < size) && (lit == -1); ind++) {
            if (coefsBis[ind].signum() != 0
                    && (!voc.isFalsified(wpb.get(ind))
                            || (voc.getLevel(wpb.get(ind)) > assertiveDL))
                    && ind != indLitImplied) {
                if (coefsBis[ind].compareTo(degreeBis) < 0) {
                    lit = ind;
                }
            }
        }

        return lit;
    }

}
