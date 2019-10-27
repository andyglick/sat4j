/*
 * This file is a part of the fr.univartois.cril.orpheus.preprocessing.degree package.
 *
 * It contains the SubsetSum, a method object used to solve instances of the subset-sum
 * problem.
 *
 * (c) Romain WALLON - Orpheus.
 * All rights reserved.
 */

package org.sat4j.pb.constraints.pb;

import java.util.BitSet;
import java.util.HashSet;

/**
 * The SubsetSum is a method object used to solve instances of the subset-sum
 * problem.
 * 
 * The computations are made using {@code int} values: {@code long} values or
 * arbitrary precision are not supported as if there is a value too big to fit
 * into an {@code int}, the resolution of subset-sum will either require too
 * much memory or time to be achieved.
 *
 * @author Romain WALLON
 *
 * @version 1.0
 */
public final class SubsetSumModulo {

    /**
     * The set of integers to find a subset-sum in.
     */
    private int[] elements;

    /**
     * The last sum that has been checked. All the preceding sums have
     * necessarily been checked.
     */
    private boolean computed = false;

    /**
     * The matrix used to compute all the subset-sums in a bottom-up manner
     * using dynamic-programming. The value of {@code allSubsetSums.get(i, j)}
     * will be {@code true} if there is a subset of at most {@code j} elements
     * with sum equal to {@code i}.
     */
    private final BitSet allSubsetSums;

    private final int modulo;

    /**
     * Creates a new SubsetSum.
     * 
     * @param maxSum
     *            The maximum value for the sum.
     * @param maxElements
     *            The maximum number of elements.
     */
    public SubsetSumModulo(int modulo) {
        this.allSubsetSums = new BitSet(modulo);
        this.modulo = modulo;
    }

    /**
     * Sets the set of integers to find a subset-sum in.
     * 
     * @param elements
     *            The set of integers to find a subset-sum in.
     */
    public void setElements(int[] elements) {
        this.elements = elements;
        if (computed) {
            computed = false;
            allSubsetSums.clear();
        }
    }

    /**
     * Checks whether there exists a subset of the associated set such that the
     * sum of its elements is equal to the given value.
     * 
     * @param sum
     *            The sum to check.
     * 
     * @return If there is a subset with a sum equal to the given value.
     */
    public boolean sumExists(int sum) {
        // Checking all the missing sums.
        if (!computed) {
            HashSet<Integer> sums = new HashSet<>();
            for (int e : elements) {
                HashSet<Integer> tmp = new HashSet<>();
                int s = mod(e);
                allSubsetSums.set(s);
                tmp.add(s);
                for (int i : sums) {
                    int is = mod(i + e);
                    allSubsetSums.set(is);
                    tmp.add(is);
                }
                sums.addAll(tmp);
            }
            computed = true;
        }
        return allSubsetSums.get(sum);
    }

    private int mod(int n) {
        int val = n % modulo;
        if (val < 0) {
            val += modulo;
        }
        return val;
    }

}
