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
import org.sat4j.pb.IPBSolver;
import org.sat4j.pb.SolverFactory;
import org.sat4j.specs.ContradictionException;
import org.sat4j.specs.IVec;
import org.sat4j.specs.IVecInt;
import org.sat4j.specs.TimeoutException;

public class ChowPostProcess implements IPostProcess {

    private static final IPostProcess INSTANCE = new ChowPostProcess();

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

    private final SubsetSum subsetSum;

    private ChowPostProcess() {
        this.subsetSum = new SubsetSum(20000, 1000);
    }

    @Override
    public void postProcess(int dl, ConflictMap conflictMap) {
        if (conflictMap.weightedLits.size() > 1000 || conflictMap.getDegree()
                .compareTo(DEFAULT_MAX_DEGREE_TO_PROCESS) > 0) {
            conflictMap.stats.numberOfConstraintsIgnoredByChow++;
            return;
        }
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
        conflictMap.stats.numberOfRemovedIrrelevantLiterals += toRemove.size();
        conflictMap.stats.maxDegreeModifiedByChow = Math.max(
                conflictMap.stats.maxDegreeModifiedByChow,
                conflictMap.degree.longValue());
        conflictMap.stats.maxSizeModifiedByChow = Math.max(
                conflictMap.stats.maxSizeModifiedByChow, conflictMap.size());
        conflictMap.stats.maxDegreeDiff = Math
                .max(conflictMap.stats.maxDegreeDiff, newDegree.longValue());
        conflictMap.stats.maxRemovedChow = Math
                .max(conflictMap.stats.maxRemovedChow, toRemove.size());

    }

    private boolean dependsOnSum(ConflictMap conflictMap, BigInteger coef,
            int litIndex) {
        int[] elts = new int[conflictMap.size() - 1];
        for (int i = 0, index = 0; i < conflictMap.size(); i++) {
            if (i != litIndex) {
                elts[index] = conflictMap.weightedLits.getCoef(i).intValue();
                index++;
            }
        }

        int degree = conflictMap.degree.intValue();
        int minSum = degree - coef.intValue();
        subsetSum.setElements(elts);

        for (int i = degree - 1; i >= minSum; i--) {
            if (subsetSum.sumExists(i)) {
                return true;
            }
        }

        return false;
    }

    private boolean dependsOn(ConflictMap conflictMap, BigInteger coef,
            int litIndex) {
        try {
            IPBSolver solver = SolverFactory.newCuttingPlanes();
            solver.setTimeout(5);
            solver.newVar(conflictMap.voc.nVars());
            IVecInt lits = new VecInt();
            IVec<BigInteger> coeffs = new Vec<>();
            for (int i = 0; i < conflictMap.size(); i++) {
                if (i != litIndex) {
                    lits.push(conflictMap.weightedLits.getLit(i));
                    coeffs.push(conflictMap.weightedLits.getCoef(i));
                }

            }
            solver.addAtMost(lits, coeffs,
                    conflictMap.degree.subtract(BigInteger.ONE));

            solver.addAtLeast(lits, coeffs, conflictMap.degree.subtract(coef));
            return solver.isSatisfiable();

        } catch (ContradictionException e) {
            return false;

        } catch (TimeoutException e) {
            return true;
        }

    }

    public static IPostProcess instance() {
        return INSTANCE;
    }

    @Override
    public String toString() {
        return "Irrelevant literals are removed from learnt constraints";
    }
}