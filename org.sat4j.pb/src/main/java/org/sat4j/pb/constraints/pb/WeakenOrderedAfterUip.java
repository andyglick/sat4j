package org.sat4j.pb.constraints.pb;

import java.math.BigInteger;

import org.sat4j.minisat.core.ILits;

public class WeakenOrderedAfterUip implements IPostUipWeakeningStrategy {

    @Override
    public int findLiteralToRemove(ILits voc, PBConstr wpb,
            BigInteger[] coefsBis, int indLitImplied, BigInteger degreeBis,
            BigInteger slack, BigInteger coeffAssertive, int assertiveDL) {
        // Any literal assigned at a level > assertiveDL may be removed
        int satlit = -1;
        int satmaxlevel = -1;
        int unassignedlit = -1;
        int falslit = -1;
        int falsmaxlevel = -1;
        int size = wpb.size();
        for (int ind = 0; (ind < size); ind++) {
            if (coefsBis[ind].signum() != 0 && ind != indLitImplied) {
                int lit = wpb.get(ind);
                if (voc.isSatisfied(lit) && voc.getLevel(lit) > satmaxlevel) {
                    satlit = ind;
                    satmaxlevel = voc.getLevel(lit);
                } else if (voc.isUnassigned(lit)) {
                    unassignedlit = ind;
                } else if (voc.isFalsified(lit)
                        && voc.getLevel(lit) > assertiveDL
                        && voc.getLevel(lit) > falsmaxlevel) {
                    falslit = ind;
                    falsmaxlevel = voc.getLevel(lit);
                }
            }
        }

        if (unassignedlit >= 0) {
            return unassignedlit;
        }

        if (satlit >= 0) {
            return satlit;
        }

        return falslit;
    }

}
