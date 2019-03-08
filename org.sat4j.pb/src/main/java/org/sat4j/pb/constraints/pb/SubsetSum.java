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
public final class SubsetSum {

    /**
     * The set of integers to find a subset-sum in.
     */
    private int[] elements;

    /**
     * The last sum that has been checked. All the preceding sums have
     * necessarily been checked.
     */
    private int lastCheckedSum;

    /**
     * The matrix used to compute all the subset-sums in a bottom-up manner
     * using dynamic-programming. The value of {@code allSubsetSums.get(i, j)}
     * will be {@code true} if there is a subset of at most {@code j} elements
     * with sum equal to {@code i}.
     */
    private final BitMatrix allSubsetSums;

    /**
     * Creates a new SubsetSum.
     * 
     * @param maxSum
     *            The maximum value for the sum.
     * @param maxElements
     *            The maximum number of elements.
     */
    public SubsetSum(int maxSum, int maxElements) {
        this.allSubsetSums = new BitMatrix(maxSum, maxElements);

        // Initializing the matrix.
        // If sum is 0, taking no element is a solution.
        for (int i = 0; i < maxElements; i++) {
            allSubsetSums.set(0, i, true);
        }
    }

    /**
     * Sets the set of integers to find a subset-sum in.
     * 
     * @param elements
     *            The set of integers to find a subset-sum in.
     */
    public void setElements(int[] elements) {
        this.elements = elements;
        this.lastCheckedSum = 0;
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
        for (int i = lastCheckedSum + 1; i <= sum; i++) {
            for (int j = 1; j <= elements.length; j++) {
                allSubsetSums.set(i, j, allSubsetSums.get(i, j - 1));

                if (i >= elements[j - 1]) {
                    allSubsetSums.set(i, j, allSubsetSums.get(i, j)
                            || allSubsetSums.get(i - elements[j - 1], j - 1));
                }
            }
        }

        // Updating the last checked sum before returning.
        lastCheckedSum = Math.max(lastCheckedSum, sum);
        return allSubsetSums.get(sum, elements.length);
    }

}
