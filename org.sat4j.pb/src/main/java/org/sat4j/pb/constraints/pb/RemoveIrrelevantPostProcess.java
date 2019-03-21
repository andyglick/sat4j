package org.sat4j.pb.constraints.pb;

import java.math.BigInteger;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map.Entry;
import java.util.NavigableMap;
import java.util.Set;
import java.util.TreeMap;

import org.sat4j.core.VecInt;
import org.sat4j.specs.IVecInt;

public class RemoveIrrelevantPostProcess implements IPostProcess {

    private static final IPostProcess INSTANCE = new RemoveIrrelevantPostProcess();

    /**
     * The default value for the maximum number of literals for the constraints
     * to process.
     */
    private static final int DEFAULT_MAX_LITERALS_TO_PROCESS = 500;

    /**
     * The default value for the maximum degree for the constraints to process.
     */
    private static final BigInteger DEFAULT_MAX_DEGREE_TO_PROCESS = BigInteger
            .valueOf(20_000);

    /**
     * The default value for the maximum degree for the constraints to process.
     */
    private static final BigInteger DEFAULT_MAX = BigInteger
            .valueOf(500_000_000l);

    private final IrrelevantLiteralDetectionStrategy irrelevantDetector = IrrelevantLiteralDetectionStrategy
            .defaultStrategy();

    private RemoveIrrelevantPostProcess() {
    }

    @Override
    public void postProcess(int dl, ConflictMap conflictMap) {
        if (conflictMap.weightedLits.size() > 1000 || conflictMap.getDegree()
                .compareTo(DEFAULT_MAX_DEGREE_TO_PROCESS) > 0) {
            conflictMap.stats.numberOfConstraintsIgnoredByChow++;
            return;
        }

        long timeBefore = System.nanoTime();

        try {
            int alit = conflictMap.weightedLits
                    .getLit(conflictMap.assertiveLiteral);

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
                coefs.computeIfAbsent(c, k -> new LinkedList<Integer>()).add(i);
                maxCoeff = maxCoeff.max(c);
            }

            if (coefs.size() == 1) {
                conflictMap.stats.numberOfNonPbConstraints++;
                return;
            }

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
                for (int i : coefs.get(c)) {
                    int lit = conflictMap.weightedLits.getLit(i);
                    toRemove.push(lit);
                    newDegree = newDegree.add(c);
                }
            }

            for (int i = 0; i < toRemove.size(); i++)
                conflictMap.removeCoef(toRemove.get(i));

            conflictMap.degree = conflictMap.degree.subtract(newDegree);

            if (smallestRelevant.compareTo(conflictMap.degree) >= 0) {
                conflictMap.stats.numberOfConstraintsWhichAreClauses++;

            } else if (smallestRelevant.equals(maxCoeff)) {
                conflictMap.stats.numberOfConstraintsWhichAreCard++;

            }

            conflictMap.saturation();

            if (maxCoeff.compareTo(conflictMap.degree) > 0) {
                conflictMap.stats.numberOfTriggeredSaturations++;
            }

            conflictMap.assertiveLiteral = conflictMap.weightedLits
                    .getFromAllLits(alit);

            conflictMap.stats.numberOfConstraintsChangedByChow++;
            conflictMap.stats.numberOfRemovedIrrelevantLiterals += toRemove
                    .size();
            conflictMap.stats.maxDegreeModifiedByChow = Math.max(
                    conflictMap.stats.maxDegreeModifiedByChow,
                    conflictMap.degree.longValue());
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

    public static IPostProcess instance() {
        return INSTANCE;
    }

    @Override
    public String toString() {
        return "Irrelevant literals are removed from learnt constraints";
    }

}