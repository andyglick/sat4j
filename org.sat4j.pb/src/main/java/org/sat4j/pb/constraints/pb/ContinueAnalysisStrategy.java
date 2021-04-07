package org.sat4j.pb.constraints.pb;

import java.math.BigInteger;

import org.sat4j.core.VecInt;
import org.sat4j.minisat.core.ILits;

public class ContinueAnalysisStrategy extends AbstractAnalysisStrategy {

    @Override
    public boolean shouldStopAfterAssertion(int currentLevel) {
        // We stop when the current level reaches the backtrack level.
        // This way, we do not need to restore decisions, as assignments are
        // removed at each resolution.
        // Said differently, instead of just canceling assignments, we (may)
        // have continued to perform resolutions for each undo.
        // Technically, continuing may still improve the backjump level,
        // but when it does not, we may have trouble restoring assignments.
        return currentLevel == assertiveDL;
    }

    @Override
    protected void resolveAfterAssertion(int litImplied, ConflictMap conflict,
            PBConstr constr) {
        int nLitImplied = litImplied ^ 1;
        if (constr == null || !conflict.weightedLits.containsKey(nLitImplied)) {
            // No resolution: undo operation should be anticipated
            undo(litImplied, conflict, nLitImplied);
            return;
        }

        int ind = 0;
        while (constr.get(ind) != litImplied) {
            ind++;
        }
        BigInteger degree = constr.getDegree();
        BigInteger[] coeffs = constr.getCoefs();
        degree = reduceUntilAssertive(conflict, litImplied, ind, coeffs, degree,
                (IWatchPb) constr);

        // A resolution is performed if and only if we can guarantee that the
        // backtrack level will not increase (i.e., get worse).
        if (degree.signum() > 0) {
            cuttingPlane(conflict, constr, degree, coeffs, ind, litImplied);

        } else {
            undo(litImplied, conflict, nLitImplied);
        }
    }

    private void undo(int litImplied, ConflictMap conflict, int nLitImplied) {
        int litLevel = ConflictMap
                .levelToIndex(conflict.voc.getLevel(litImplied));
        int lit = 0;
        if (conflict.byLevel[litLevel] != null) {
            if (conflict.byLevel[litLevel].contains(litImplied)) {
                lit = litImplied;
                assert conflict.weightedLits.containsKey(litImplied);
            } else if (conflict.byLevel[litLevel].contains(nLitImplied)) {
                lit = nLitImplied;
                assert conflict.weightedLits.containsKey(nLitImplied);
            }
        }

        if (lit > 0) {
            conflict.byLevel[litLevel].remove(lit);
            if (conflict.byLevel[0] == null) {
                conflict.byLevel[0] = new VecInt();
            }
            conflict.byLevel[0].push(lit);
        }
    }

    private BigInteger reduceUntilAssertive(ConflictMap conflict,
            int litImplied, int ind, BigInteger[] reducedCoefs,
            BigInteger degreeReduced, IWatchPb wpb) {
        BigInteger slackResolve = BigInteger.ONE.negate();
        BigInteger slackThis = BigInteger.ZERO;
        BigInteger slackConflict = conflict.computeSlack(assertiveDL + 1);
        BigInteger reducedDegree = degreeReduced;
        BigInteger previousCoefLitImplied = BigInteger.ZERO;
        BigInteger coefLitImplied = conflict.weightedLits.get(litImplied ^ 1);
        conflict.possReducedCoefs = slackConstraint(conflict.voc, wpb,
                reducedCoefs);

        do {
            if (slackResolve.compareTo(
                    conflict.weightedLits.getCoef(assertiveLitIndex)) >= 0) {
                // Weakening a literal to (try) to preserve the assertion level.
                assert slackThis.signum() > 0;
                BigInteger tmp = reduceInConstraint(conflict, wpb, reducedCoefs,
                        ind, reducedDegree, slackResolve);
                if (tmp.signum() == 0) {
                    // Then we do not resolve because we cannot preserve the
                    // assertion level.
                    return tmp;
                }
                reducedDegree = tmp;
            }

            // search of the multiplying coefficients
            assert conflict.weightedLits.get(litImplied ^ 1).signum() > 0;
            assert reducedCoefs[ind].signum() > 0;

            if (!reducedCoefs[ind].equals(previousCoefLitImplied)) {
                assert coefLitImplied
                        .equals(conflict.weightedLits.get(litImplied ^ 1));
                BigInteger ppcm = conflict.ppcm(reducedCoefs[ind],
                        coefLitImplied);
                assert ppcm.signum() > 0;
                conflict.coefMult = ppcm.divide(coefLitImplied);
                conflict.coefMultCons = ppcm.divide(reducedCoefs[ind]);

                assert conflict.coefMultCons.signum() > 0;
                assert conflict.coefMult.signum() > 0;
                assert conflict.coefMult.multiply(coefLitImplied).equals(
                        conflict.coefMultCons.multiply(reducedCoefs[ind]));
                previousCoefLitImplied = reducedCoefs[ind];
            }

            // slacks computed for each constraint
            slackThis = conflict.possReducedCoefs.subtract(reducedDegree)
                    .multiply(conflict.coefMultCons);
            assert slackThis
                    .equals(slackConstraint(conflict.voc, wpb, reducedCoefs)
                            .subtract(reducedDegree)
                            .multiply(conflict.coefMultCons));
            assert slackConflict.equals(conflict.computeSlack(assertiveDL + 1));
            BigInteger slackIndex = slackConflict.multiply(conflict.coefMult);
            // assert slackIndex.compareTo();
            // TODO Really update the slack, not only estimate.
            slackResolve = slackThis.add(slackIndex);
        } while ((slackResolve.compareTo(
                conflict.weightedLits.getCoef(assertiveLitIndex)) >= 0)
                || conflict.isUnsat());

        assert conflict.coefMult
                .multiply(conflict.weightedLits.get(litImplied ^ 1))
                .equals(conflict.coefMultCons.multiply(reducedCoefs[ind]));

        return reducedDegree;

    }

    public BigInteger reduceInConstraint(ConflictMap conflict, IWatchPb wpb,
            final BigInteger[] coefsBis, final int indLitImplied,
            final BigInteger degreeBis, BigInteger slackResolve) {
        assert degreeBis.compareTo(BigInteger.ONE) > 0;

        // search of a literal to remove
        int lit = findLiteralToRemove(conflict.voc, wpb, coefsBis,
                indLitImplied, degreeBis);

        // If no literal has been found, we do not resolve.
        // Note that this is possible as we already know that we will propagate
        // at some point.
        if (lit < 0 || lit == indLitImplied) {
            return BigInteger.ZERO;
        }

        // Reduction can be done
        BigInteger degUpdate = degreeBis.subtract(coefsBis[lit]);
        conflict.possReducedCoefs = conflict.possReducedCoefs
                .subtract(coefsBis[lit]);
        coefsBis[lit] = BigInteger.ZERO;
        assert conflict.possReducedCoefs
                .equals(slackConstraint(conflict.voc, wpb, coefsBis));

        // saturation of the constraint
        degUpdate = conflict.saturation(coefsBis, degUpdate, wpb);

        assert coefsBis[indLitImplied].signum() > 0;
        assert degreeBis.compareTo(degUpdate) > 0;
        assert conflict.possReducedCoefs
                .equals(slackConstraint(conflict.voc, wpb, coefsBis));
        return degUpdate;
    }

    private int findLiteralToRemove(ILits voc, IWatchPb wpb,
            BigInteger[] coefsBis, int indLitImplied, BigInteger degreeBis) {
        // Any level > assertiveDL may be removed
        int lit = -1;
        int size = wpb.size();
        for (int ind = 0; ind < size && lit == -1; ind++) {
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

    private BigInteger slackConstraint(ILits voc, PBConstr wpb,
            BigInteger[] coeffs) {
        BigInteger slack = BigInteger.ZERO;
        for (int i = 0; i < wpb.size(); i++) {
            BigInteger tmp = coeffs[i];
            int lit = wpb.get(i);
            if (tmp.signum() > 0 && (!voc.isFalsified(lit)
                    || voc.getLevel(lit) > assertiveDL)) {
                slack = slack.add(tmp);
            }
        }
        return slack;
    }

    private void cuttingPlane(ConflictMap conflict, PBConstr constr,
            BigInteger degreeCons, BigInteger[] coefsCons, int indLitInConstr,
            int litImplied) {
        if (!conflict.coefMult.equals(BigInteger.ONE)) {
            for (int i = 0; i < conflict.size(); i++) {
                conflict.changeCoef(i, conflict.weightedLits.getCoef(i)
                        .multiply(conflict.coefMult));
            }
        }
        // TODO Really here??
        conflict.degree = conflict.degree.multiply(conflict.coefMult);

        // cutting plane
        conflict.degree = conflict.cuttingPlane(constr, degreeCons, coefsCons,
                conflict.coefMultCons, solver, indLitInConstr);

        // neither litImplied nor nLitImplied is present in coefs structure
        assert !conflict.weightedLits.containsKey(litImplied);
        assert !conflict.weightedLits.containsKey(litImplied ^ 1);
        // neither litImplied nor nLitImplied is present in byLevel structure
        assert conflict.getLevelByLevel(litImplied) == -1;
        assert conflict.getLevelByLevel(litImplied ^ 1) == -1;
        assert conflict.degree.signum() > 0;
        assert conflict.computeSlack(assertiveDL).subtract(conflict.degree)
                .compareTo(
                        conflict.weightedLits.getCoef(assertiveLitIndex)) < 0;
        // saturation
        conflict.degree = conflict.saturation();
        assert conflict.computeSlack(assertiveDL).subtract(conflict.degree)
                .compareTo(
                        conflict.weightedLits.getCoef(assertiveLitIndex)) < 0;
        conflict.divideCoefs();
    }

}
