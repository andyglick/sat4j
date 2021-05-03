package org.sat4j.pb.constraints.pb;

import java.math.BigInteger;
import java.util.HashMap;
import java.util.Map;

import org.sat4j.core.VecInt;

/**
 * The ContinueAnalysisStrategy implements the analysis strategy that allows to
 * continue the analysis as long as this preserves the assertion.
 *
 * @author Romain Wallon
 */
public class ContinueAnalysisStrategy extends AbstractAnalysisStrategy {

    private boolean previousConflicting = false;

    private final IStopAnalysisStrategy stopStrategy;

    private final IPostUipWeakeningStrategy weakeningStrategy;

    private ContinueAnalysisStrategy(IStopAnalysisStrategy stopStrategy,
            IPostUipWeakeningStrategy weakeningStrategy) {
        this.stopStrategy = stopStrategy;
        this.weakeningStrategy = weakeningStrategy;
    }

    public static ContinueAnalysisStrategy newContinueUntilBackjumpLevel(
            IPostUipWeakeningStrategy weakeningStrategy) {
        // We stop when the current level reaches the backtrack level.
        // This way, we do not need to restore decisions, as assignments are
        // removed at each resolution.
        // Said differently, instead of just canceling assignments, we (may)
        // have continued to perform resolutions for each undo.
        // Technically, continuing may still improve the backjump level,
        // but when it does not, we may have trouble restoring assignments.
        return new ContinueAnalysisStrategy((a, c) -> c <= a,
                weakeningStrategy);
    }

    public static ContinueAnalysisStrategy newContinueUnlessTopLevel(
            IPostUipWeakeningStrategy weakeningStrategy) {
        // We stop when the backjump level is 0, as it cannot be improved.
        return new ContinueAnalysisStrategy((a, c) -> c <= a || a == 0,
                weakeningStrategy);
    }

    public static ContinueAnalysisStrategy newContinueUnlessHighLevel(
            IPostUipWeakeningStrategy weakeningStrategy) {
        // We stop when the backjump level is "high enough"
        return new ContinueAnalysisStrategy(
                (a, c) -> c <= a || ((c > 0) && (100 * a / c) <= 10),
                weakeningStrategy);
    }

    @Override
    public void isAssertiveAt(int dl, int assertiveLit) {
        if (dl != assertiveDL) {
            previousConflicting = false;
        }
        super.isAssertiveAt(dl, assertiveLit);
    }

    @Override
    public boolean shouldStopAfterAssertion(int currentLevel,
            ConflictMap conflict) {
        return stopStrategy.shouldStop(assertiveDL, currentLevel)
                && conflict.propagatesNow(currentLevel)
                && conflict.slackConflict().signum() >= 0;
    }

    @Override
    protected void resolveAfterAssertion(int litImplied, ConflictMap conflict,
            PBConstr constr) {
        int nLitImplied = litImplied ^ 1;
        if (constr == null || !conflict.weightedLits.containsKey(nLitImplied)) {
            // No canceling candidate: undo operation should be anticipated
            undo(litImplied, conflict, nLitImplied);
            return;
        }

        // Looking for the literal to remove.
        int ind = 0;
        while (constr.get(ind) != litImplied) {
            ind++;
        }

        // Weakening the reason constraint to preserve the assertion.
        // Note that this is different from only preserving the conflict.
        // Hence, we need dedicated operations.
        BigInteger degree = constr.getDegree();
        BigInteger[] coeffs = constr.getCoefs();
        degree = reduceUntilAssertive(conflict, litImplied, ind, coeffs, degree,
                constr);

        // A resolution is performed if and only if we can guarantee that the
        // backtrack level will not increase (i.e., get worse).
        // This is guaranteed only when reduceUntilAssertive return a value > 0.
        if (degree.signum() > 0) {
            cuttingPlane(conflict, constr, degree, coeffs, ind, litImplied);

        } else {
            undo(litImplied, conflict, nLitImplied);
        }
    }

    private BigInteger reduceUntilAssertive(ConflictMap conflict,
            int litImplied, int ind, BigInteger[] reducedCoefs,
            BigInteger degreeReduced, PBConstr wpb) {
        BigInteger reducedDegree = degreeReduced;
        BigInteger previousCoefLitImplied = BigInteger.ZERO;
        BigInteger coefLitImplied = conflict.weightedLits.get(litImplied ^ 1);
        BigInteger coeffAssertive = BigInteger.ZERO;
        BigInteger finalSlack = BigInteger.ONE.negate();

        do {

            if (finalSlack.compareTo(coeffAssertive) >= 0) {
                // Weakening a literal to (try) to preserve the assertion level.
                BigInteger tmp = reduceInConstraint(conflict, wpb, reducedCoefs,
                        ind, reducedDegree, finalSlack, coeffAssertive);
                if (tmp.signum() == 0) {
                    // Then we do not resolve because we cannot preserve the
                    // assertion level.
                    return tmp;
                }
                reducedDegree = tmp;
            }

            // Search of the multiplying coefficients
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

            BigInteger[] slackAndCoeffAndDegree = computeSlack(conflict, wpb,
                    reducedDegree, reducedCoefs);
            finalSlack = slackAndCoeffAndDegree[0];
            coeffAssertive = slackAndCoeffAndDegree[1];
            if (slackAndCoeffAndDegree[2].signum() <= 0) {
                return BigInteger.ZERO;
            }
        } while ((finalSlack.compareTo(coeffAssertive) >= 0)
                || conflict.isUnsat());

        assert conflict.coefMult
                .multiply(conflict.weightedLits.get(litImplied ^ 1))
                .equals(conflict.coefMultCons.multiply(reducedCoefs[ind]));
        previousConflicting = finalSlack.signum() < 0;

        return reducedDegree;

    }

    public BigInteger reduceInConstraint(ConflictMap conflict, PBConstr wpb,
            final BigInteger[] coefsBis, final int indLitImplied,
            final BigInteger degreeBis, BigInteger slack,
            BigInteger coeffAssertive) {
        if (degreeBis.signum() <= 0) {
            return BigInteger.ZERO;
        }

        // Search for a literal to remove.
        int lit = weakeningStrategy.findLiteralToRemove(conflict.voc, wpb,
                coefsBis, indLitImplied, degreeBis, slack, coeffAssertive,
                assertiveDL);

        // If no literal has been found, we do not resolve.
        // Note that this is possible as we already know that we will propagate
        // at some point.
        if ((lit < 0) || (lit == indLitImplied)) {
            return BigInteger.ZERO;
        }

        // Weakening the chosen literal.
        BigInteger coeff = coefsBis[lit];
        coefsBis[lit] = BigInteger.ZERO;
        BigInteger degUpdate = degreeBis.subtract(coeff);

        // Saturation of the constraint
        degUpdate = conflict.saturation(coefsBis, degUpdate, wpb);

        assert coefsBis[indLitImplied].signum() > 0;
        assert degreeBis.compareTo(degUpdate) > 0;
        return degUpdate;
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

        // TODO Check if this is really the good place to do this operation.
        conflict.degree = conflict.degree.multiply(conflict.coefMult);
        conflict.degree = conflict.cuttingPlane(constr,
                degreeCons.multiply(conflict.coefMultCons), coefsCons,
                conflict.coefMultCons, solver, indLitInConstr);

        // neither litImplied nor nLitImplied is present in coefs structure
        assert !conflict.weightedLits.containsKey(litImplied);
        assert !conflict.weightedLits.containsKey(litImplied ^ 1);
        // neither litImplied nor nLitImplied is present in byLevel structure
        assert conflict.getLevelByLevel(litImplied) == -1;
        assert conflict.getLevelByLevel(litImplied ^ 1) == -1;

        assert conflict.degree.signum() > 0;

        // saturation
        conflict.degree = conflict.saturation();
        assert checkAssertionLevel(conflict) : "Conflict is not assertive at "
                + assertiveDL + ": " + conflict.toString();
        conflict.divideCoefs();

        // Collecting statistics
        if (firstCancellationAfterAssertion) {
            conflict.stats.incNbSubOptimalAnalyses();
            firstCancellationAfterAssertion = false;
        }
        conflict.stats.incNbResolutionsAfterAssertion();

    }

    private boolean checkAssertionLevel(ConflictMap conflict) {
        BigInteger slack = conflict.computeSlack(assertiveDL + 1)
                .subtract(conflict.degree);

        for (int i = 0; i < conflict.size(); i++) {
            int lit = conflict.weightedLits.getLit(i);
            if (!conflict.voc.isFalsified(lit)
                    || conflict.voc.getLevel(lit) > assertiveDL) {
                if (conflict.weightedLits.getCoef(i).compareTo(slack) > 0) {
                    return true;
                }
            }
        }
        return slack.signum() <= 0;
    }

    private void undo(int litImplied, ConflictMap conflict, int nLitImplied) {
        // Looking for the (potential) occurrence of the literal in the
        // conflict.
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

        // Removing the assignment of this literal.
        if (lit > 0) {
            conflict.byLevel[litLevel].remove(lit);
            if (conflict.byLevel[0] == null) {
                conflict.byLevel[0] = new VecInt();
            }
            conflict.byLevel[0].push(lit);
        }
    }

    public BigInteger[] computeSlack(ConflictMap conflict, PBConstr cpb,
            BigInteger degreeCons, BigInteger[] reducedCoefs) {
        BigInteger degree = conflict.degree.multiply(conflict.coefMult)
                .add(degreeCons.multiply(conflict.coefMultCons));
        Map<Integer, BigInteger> litCoef = new HashMap<Integer, BigInteger>();
        for (int i = 0; i < cpb.size(); i++) {
            BigInteger coef = reducedCoefs[i];
            int lit = cpb.get(i);
            int nlit = lit ^ 1;
            if (coef.signum() > 0) {
                coef = coef.multiply(conflict.coefMultCons);
                if (conflict.weightedLits.containsKey(nlit)) {
                    BigInteger tmp = conflict.weightedLits.get(nlit)
                            .multiply(conflict.coefMult);
                    if (tmp.compareTo(coef) < 0) {
                        litCoef.put(lit, coef.subtract(tmp));
                        degree = degree.subtract(tmp);

                    } else {
                        if (tmp.equals(coef)) {
                            litCoef.put(lit, BigInteger.ZERO);
                            degree = degree.subtract(coef);

                        } else {
                            litCoef.put(nlit, tmp.subtract(coef));
                            degree = degree.subtract(coef);
                        }
                    }
                } else {
                    if (conflict.weightedLits.containsKey(lit)) {
                        litCoef.put(lit, conflict.weightedLits.get(lit)
                                .multiply(conflict.coefMult).add(coef));

                    } else {
                        litCoef.put(lit, coef);
                    }
                }
            }
        }

        // Adding missing literals.
        for (int l = 0; l < conflict.size(); l++) {
            int lit = conflict.weightedLits.getLit(l);
            if (!litCoef.containsKey(lit) && !litCoef.containsKey(lit ^ 1)) {
                litCoef.put(lit, conflict.weightedLits.get(lit)
                        .multiply(conflict.coefMult));
            }
        }

        // Computing the slack.
        BigInteger slack = BigInteger.ZERO;
        BigInteger max = BigInteger.ZERO;
        for (Map.Entry<Integer, BigInteger> wl : litCoef.entrySet()) {
            if (!conflict.voc.isFalsified(wl.getKey())
                    || conflict.voc.getLevel(wl.getKey()) > assertiveDL) {
                slack = slack.add(wl.getValue().min(degree));
            }

            if (conflict.voc.getLevel(wl.getKey()) > assertiveDL) {
                max = max.max(wl.getValue().min(degree));
            }
        }
        slack = slack.subtract(degree);

        return new BigInteger[] { slack, max, degree };
    }

    @Override
    public int getBacktrackLevel(ConflictMap confl, int currentLevel) {
        return Math.min(assertiveDL, currentLevel);
    }

    @Override
    public void undoOne(ConflictMap confl, int last) {
    }
}
