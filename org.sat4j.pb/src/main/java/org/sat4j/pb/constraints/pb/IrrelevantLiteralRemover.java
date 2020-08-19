package org.sat4j.pb.constraints.pb;

import java.math.BigInteger;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map.Entry;
import java.util.NavigableMap;
import java.util.Set;
import java.util.TreeMap;

import org.sat4j.core.Vec;
import org.sat4j.core.VecInt;
import org.sat4j.pb.core.PBSolverStats;
import org.sat4j.specs.IVecInt;

public class IrrelevantLiteralRemover
        implements IPostProcess, IIrrelevantLiteralRemover {

    private static final IrrelevantLiteralRemover INSTANCE = new IrrelevantLiteralRemover();

    /**
     * The default value for the maximum number of literals for the constraints
     * to process.
     */
    static final int MAX_LITERALS = 500;

    private final IrrelevantLiteralDetectionStrategy irrelevantDetector = IrrelevantLiteralDetectionStrategyFactory
            .defaultStrategy();

    private IrrelevantLiteralRemover() {
    }

    public static IrrelevantLiteralRemover instance() {
        return INSTANCE;
    }

    @Override
    public void postProcess(int dl, ConflictMap conflictMap) {
        if (conflictMap.weightedLits.size() > MAX_LITERALS) {
            conflictMap.stats.numberOfConstraintsIgnoredByChow++;
            return;
        }

        long timeBefore = System.nanoTime();

        try {
            int alit = -1;
            int lvl = dl;
            BigInteger coeff = null;
            if (conflictMap.assertiveLiteral >= 0) {
                alit = conflictMap.weightedLits
                        .getLit(conflictMap.assertiveLiteral);
                lvl = conflictMap.getBacktrackLevel(dl);
                coeff = conflictMap.weightedLits
                        .getCoef(conflictMap.assertiveLiteral);
            }
            conflictMap.stats.maxDegreeForChow = Math.max(
                    conflictMap.stats.maxDegreeForChow,
                    conflictMap.degree.longValue());
            conflictMap.stats.maxSizeForChow = Math
                    .max(conflictMap.stats.maxSizeForChow, conflictMap.size());

            NavigableMap<BigInteger, List<Integer>> coefs = new TreeMap<>();
            BigInteger maxCoeff = BigInteger.ZERO;
            for (int i = 0; i < conflictMap.size(); i++) {
                BigInteger c = conflictMap.weightedLits.getCoef(i)
                        .min(conflictMap.degree);
                List<Integer> l = coefs.get(c);
                if (l == null) {
                    l = new LinkedList<Integer>();
                    coefs.put(c, l);
                }
                l.add(i);
                maxCoeff = maxCoeff.max(c);
            }

            if (coefs.size() == 1) {
                conflictMap.stats.numberOfNonPbConstraints++;
                return;
            }
            BigInteger slack = conflictMap.computeSlack(lvl + 1)
                    .subtract(conflictMap.degree);
            Set<BigInteger> irrelevant = new HashSet<>();
            BigInteger smallestRelevant = BigInteger.ZERO;
            for (Entry<BigInteger, List<Integer>> e : coefs.entrySet()) {
                if (dependsOn(conflictMap, e.getKey(), e.getValue().get(0))) {
                    smallestRelevant = e.getKey();
                    break;
                }
                irrelevant.add(e.getKey());
            }

            if (irrelevant.isEmpty()) {
                conflictMap.stats.numberOfConstraintsNotChangedByChow++;
                return;
            }

            IVecInt toRemove = new VecInt();
            BigInteger newDegree = BigInteger.ZERO;
            for (BigInteger c : irrelevant) {
                BigInteger allRemoved = c
                        .multiply(BigInteger.valueOf(coefs.get(c).size()));
                BigInteger newSlack = slack.add(allRemoved);
                if (newSlack.signum() >= 0
                        || (coeff != null && coeff.compareTo(newSlack) <= 0)) {
                    // The conflict or assertion will not be preserved if we
                    // weaken here.
                    continue;
                }
                slack = newSlack;
                newDegree = newDegree.add(allRemoved);
                for (int i : coefs.get(c)) {
                    int lit = conflictMap.weightedLits.getLit(i);
                    toRemove.push(lit);
                }
            }

            for (int i = 0; i < toRemove.size(); i++)
                conflictMap.removeCoef(toRemove.get(i));

            BigInteger reducedDegree = conflictMap.degree.subtract(newDegree);
            BigInteger sumAllCoef = BigInteger.ZERO;
            BigInteger sumAllCoefSaturated = BigInteger.ZERO;

            for (int i = 0; i < conflictMap.size(); i++) {
                sumAllCoef = sumAllCoef.add(conflictMap.weightedLits.getCoef(i)
                        .min(conflictMap.degree));
                sumAllCoefSaturated = sumAllCoefSaturated.add(
                        conflictMap.weightedLits.getCoef(i).min(reducedDegree));
            }
            if (sumAllCoef.subtract(conflictMap.degree).compareTo(
                    sumAllCoefSaturated.subtract(reducedDegree)) > 0) {
                conflictMap.degree = reducedDegree;
                conflictMap.stats.numberOfTriggeredSaturations++;
            }
            if (smallestRelevant.compareTo(conflictMap.degree) >= 0) {
                conflictMap.stats.numberOfConstraintsWhichAreClauses++;

            } else if (smallestRelevant.equals(maxCoeff)) {
                conflictMap.stats.numberOfConstraintsWhichAreCard++;

            }

            conflictMap.saturation();
            if (alit >= 0) {
                conflictMap.assertiveLiteral = conflictMap.weightedLits
                        .getFromAllLits(alit);
            }
            conflictMap.stats.numberOfConstraintsChangedByChow++;
            conflictMap.stats.numberOfRemovedIrrelevantLiterals += toRemove
                    .size();
            conflictMap.stats.numberOfRemovedIrrelevantLiteralsAfterCancellation += toRemove
                    .size();
            conflictMap.stats.maxDegreeModifiedByChow = Math.max(
                    conflictMap.stats.maxDegreeModifiedByChow,
                    conflictMap.degree.add(newDegree).longValue());
            conflictMap.stats.maxSizeModifiedByChow = Math.max(
                    conflictMap.stats.maxSizeModifiedByChow,
                    conflictMap.size());
            conflictMap.stats.maxDegreeDiff = Math.max(
                    conflictMap.stats.maxDegreeDiff, newDegree.longValue());
            conflictMap.stats.maxRemovedChow = Math
                    .max(conflictMap.stats.maxRemovedChow, toRemove.size());
        } finally {
            conflictMap.stats.timeSpentDetectingIrrelevant += System.nanoTime()
                    - timeBefore;
        }

    }

    private boolean dependsOn(ConflictMap conflictMap, BigInteger coef,
            int litIndex) {
        return irrelevantDetector.dependsOn(conflictMap.voc.nVars(),
                conflictMap.weightedLits.getLits(),
                conflictMap.weightedLits.getCoefs(), conflictMap.degree,
                litIndex, coef);
    }

    private boolean dependsOn(int nVars, IVecInt literals,
            BigInteger[] coefficients, BigInteger degree, BigInteger coef,
            int litIndex) {
        return irrelevantDetector.dependsOn(nVars, literals,
                Vec.of(coefficients), degree, litIndex, coef);
    }

    @Override
    public BigInteger remove(int nVars, int[] literals,
            BigInteger[] coefficients, BigInteger degree, PBSolverStats stats,
            boolean conflict) {
        if (literals.length > MAX_LITERALS) {
            stats.numberOfConstraintsIgnoredByChow++;
            return degree;
        }

        long timeBefore = System.nanoTime();

        try {
            stats.maxDegreeForChow = Math.max(stats.maxDegreeForChow,
                    degree.longValue());
            stats.maxSizeForChow = Math.max(stats.maxSizeForChow,
                    literals.length);

            NavigableMap<BigInteger, List<Integer>> coefs = new TreeMap<>();
            BigInteger maxCoeff = BigInteger.ZERO;
            for (int i = 0; i < literals.length; i++) {
                BigInteger c = coefficients[i];
                if (c.signum() == 0) {
                    // The literals has been removed.
                    continue;
                }

                List<Integer> l = coefs.get(c);
                if (l == null) {
                    l = new LinkedList<Integer>();
                    coefs.put(c, l);
                }
                l.add(i);
                maxCoeff = maxCoeff.max(c);
            }

            if (coefs.size() == 1) {
                stats.numberOfNonPbConstraints++;
                return degree;
            }
            int nRemoved = 0;
            Set<BigInteger> irrelevant = new HashSet<>();
            BigInteger smallestRelevant = BigInteger.ZERO;
            for (Entry<BigInteger, List<Integer>> e : coefs.entrySet()) {
                if (dependsOn(nVars, VecInt.of(literals), coefficients, degree,
                        e.getKey(), e.getValue().get(0))) {
                    smallestRelevant = e.getKey();
                    break;
                }
                irrelevant.add(e.getKey());
            }

            if (irrelevant.isEmpty()) {
                stats.numberOfConstraintsNotChangedByChow++;
                return degree;
            }

            BigInteger newDegree = BigInteger.ZERO;
            for (BigInteger c : irrelevant) {
                BigInteger allRemoved = c
                        .multiply(BigInteger.valueOf(coefs.get(c).size()));
                newDegree = newDegree.add(allRemoved);
                for (int i : coefs.get(c)) {
                    coefficients[i] = BigInteger.ZERO;
                    nRemoved++;
                }
            }

            BigInteger reducedDegree = degree.subtract(newDegree);
            BigInteger sumAllCoef = BigInteger.ZERO;
            BigInteger sumAllCoefSaturated = BigInteger.ZERO;

            for (int i = 0; i < coefficients.length; i++) {
                sumAllCoef = sumAllCoef.add(coefficients[i].min(degree));
                sumAllCoefSaturated = sumAllCoefSaturated
                        .add(coefficients[i].min(reducedDegree));
            }
            if (sumAllCoef.subtract(degree).compareTo(
                    sumAllCoefSaturated.subtract(reducedDegree)) > 0) {
                degree = reducedDegree;
                stats.numberOfTriggeredSaturations++;
            }
            if (smallestRelevant.compareTo(degree) >= 0) {
                stats.numberOfConstraintsWhichAreClauses++;

            } else if (smallestRelevant.equals(maxCoeff)) {
                stats.numberOfConstraintsWhichAreCard++;

            }

            stats.numberOfConstraintsChangedByChow++;
            stats.numberOfRemovedIrrelevantLiterals += nRemoved;
            if (conflict)
                stats.numberOfRemovedIrrelevantLiteralsAfterWeakeningConflict += nRemoved;
            else
                stats.numberOfRemovedIrrelevantLiteralsAfterWeakeningReason += nRemoved;
            stats.maxDegreeModifiedByChow = Math.max(
                    stats.maxDegreeModifiedByChow,
                    degree.add(newDegree).longValue());
            stats.maxSizeModifiedByChow = Math.max(stats.maxSizeModifiedByChow,
                    literals.length);
            stats.maxDegreeDiff = Math.max(stats.maxDegreeDiff,
                    newDegree.longValue());
            stats.maxRemovedChow = Math.max(stats.maxRemovedChow, nRemoved);
            return degree;

        } finally {
            stats.timeSpentDetectingIrrelevant += System.nanoTime()
                    - timeBefore;
        }

    }

    @Override
    public String toString() {
        return "Irrelevant literals are removed from derived constraints";
    }

}