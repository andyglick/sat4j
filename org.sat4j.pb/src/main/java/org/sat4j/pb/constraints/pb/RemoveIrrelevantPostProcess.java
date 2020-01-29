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
    static final int MAX_LITERALS = 500;

    private final IrrelevantLiteralDetectionStrategy irrelevantDetector = IrrelevantLiteralDetectionStrategyFactory
            .defaultStrategy();

    private RemoveIrrelevantPostProcess() {
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
                System.out.println("saturation");
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

    public static IPostProcess instance() {
        return INSTANCE;
    }

    @Override
    public String toString() {
        return "Irrelevant literals are removed from learnt constraints";
    }

}