/*
 * This file is a part of the fr.univartois.cril.orpheus.preprocessing.degree package.
 *
 * It contains the 
 *
 * (c) Romain WALLON - Orpheus.
 * All rights reserved.
 */

package org.sat4j.pb.constraints.pb;

import java.math.BigInteger;

/**
 *
 * @author Romain WALLON
 *
 * @version 1.0
 */
public final class Knapsack {

    /**
     * The maximum value for the elements which are considered.
     */
    private final int maxTotalValue;

    /**
     * The values of the objects.
     */
    private int[] values;

    /**
     * The maximum value of the current elements.
     */
    private int maxValue;

    /**
     * The weight of the objects.
     */
    private BigInteger[] weights;

    /**
     * The weights that have been computed w.r.t. the used objects.
     */
    private final BigInteger[][] computedWeights;

    /**
     * Creates a new Knapsack solver.
     * 
     * @param maxValue
     *            The maximum value for the elements which are considered.
     * @param maxElem
     *            The maximum number of elements.
     */
    public Knapsack(int maxValue, int maxElem) {
        this.maxTotalValue = maxElem * maxValue;
        this.computedWeights = new BigInteger[maxElem + 1][maxTotalValue + 1];

        this.computedWeights[0][0] = BigInteger.ZERO;
    }

    /**
     * Sets the values of the elements.
     */
    public void setValues(int[] values, BigInteger[] weights) {
        this.maxValue = 0;
        this.values = values;
        this.weights = weights;

        for (int v : values) {
            maxValue = Math.max(maxValue, v);
        }
    }

    /**
     */
    public int bestValue(BigInteger capacity) {
        int max = values.length * maxValue;
        for (int i = 1; i < values.length; i++) {
            for (int x = 0; x <= max; x++) {
                BigInteger newWeight = weights[i - 1];
                if (values[i - 1] < x) {
                    newWeight = newWeight
                            .add(computedWeights[i - 1][x - values[i - 1]]);
                }
                computedWeights[i][x] = min(computedWeights[i - 1][x],
                        newWeight);
            }
        }

        int best = 0;
        int i = values.length;
        for (int x = 0; x <= max; x++) {
            BigInteger newWeight = weights[i - 1];
            if (values[i - 1] < x) {
                newWeight = newWeight
                        .add(computedWeights[i - 1][x - values[i - 1]]);
            }
            computedWeights[i][x] = min(computedWeights[i - 1][x], newWeight);
            if (computedWeights[i][x].compareTo(capacity) <= 0) {
                best = x;
            }
        }
        return best;
    }

    /**
     * Computes the minimum of two {@link BigInteger}, for which a {@code null}
     * value means {@code +Infinity}.
     * 
     * @param a
     * @param b
     * 
     * @return
     */
    private static BigInteger min(BigInteger a, BigInteger b) {
        if (a == null) {
            return b;
        }
        if (b == null) {
            return a;
        }
        return a.min(b);
    }

}
