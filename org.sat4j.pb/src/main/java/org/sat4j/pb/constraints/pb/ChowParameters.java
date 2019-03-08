/*
 * This file is a part of the fr.univartois.cril.orpheus.preprocessing.chow package.
 *
 * It contains the ChowParameters, which computes and manages the Chow parameters of a
 * pseudo-Boolean constraint.
 *
 * (c) Romain WALLON - Orpheus.
 * All rights reserved.
 */

package org.sat4j.pb.constraints.pb;

import java.math.BigInteger;
import java.util.Arrays;

import org.sat4j.specs.IVec;
import org.sat4j.specs.IVecInt;

/**
 * The ChowParameters computes and manages the Chow parameters of a
 * pseudo-Boolean constraint.
 *
 * The computation is made using {@code int} values: {@code long} values or
 * arbitrary precision are not supported as if there is a value too big to fit
 * into an {@code int}, the computation of the Chow parameters will either
 * require too much memory or time to be achieved.
 * 
 * @author Romain WALLON
 *
 * @version 1.0
 */
final class ChowParameters {

    private IVecInt literals;

    private IVec<BigInteger> coefficients;

    private BigInteger degree;

    private int size;

    /**
     * The Chow parameters of the constraint.
     */
    private final BigInteger[] parameters;

    /**
     * The modified Chow parameters of the constraint.
     */
    private final BigInteger[] modifiedParameters;

    /**
     * The array telling which parameters have already been computed.
     */
    private final boolean[] computedParameters;

    /**
     * The array helping to compute the number of true points of a constraint.
     * It is filled using dynamic programming. Note that
     * {@code falsePoints[i][d]} is equal to the number of counter-models of the
     * considered constraints with at most {@code i} literals satisfied
     * realizing a sum at most equal to {@code d}.
     */
    private final BigInteger[][] falsePoints;

    /**
     * Creates a new ChowParameters.
     * 
     * @param maxLiterals
     *            The maximum number of literals for a constraint.
     * @param maxDegree
     *            The maximum degree for a constraint.
     */
    ChowParameters(int maxLiterals, int maxDegree) {
        this.parameters = new BigInteger[maxLiterals + 1];
        this.modifiedParameters = new BigInteger[maxLiterals + 1];
        this.computedParameters = new boolean[maxLiterals + 1];
        this.falsePoints = new BigInteger[maxLiterals + 1][maxDegree];
        Arrays.fill(falsePoints[0], BigInteger.ZERO);
    }

    /**
     * 
     */
    void reset() {
        Arrays.fill(computedParameters, false);
    }

    /**
     * Uses the modified Chow parameters of the associated constraint to check
     * whether it depends on the literal at the given index.
     * 
     * @param index
     *            The index of the literal to check.
     * 
     * @return If the constraint depends on the literal.
     */
    boolean dependsOn(int index) {
        return getModifiedChowParameter(index).signum() != 0;
    }

    /**
     * Uses the modified Chow parameters of the associated constraint to compare
     * two literals in the constraint to check whether they are equivalent, or
     * if one of them is <i>preferred</i> to the other (in terms of
     * coefficients).
     * 
     * @param first
     *            The index of the first literal to compare.
     * @param second
     *            The index of the second literal to compare.
     * 
     * @return The value {@code 0} if both literals are equivalent; a value less
     *         than {@code 0} if {@code second} is preferred to {@code first}; a
     *         value greater than {@code 0} if {@code first} is preferred to
     *         {@code second}.
     */
    int compare(int first, int second) {
        return getModifiedChowParameter(first)
                .compareTo(getModifiedChowParameter(second));
    }

    /**
     * Gives the Chow parameter at the given index. If its value has not been
     * computed yet, invoking this method will compute it. Otherwise, the cached
     * value is returned.
     * 
     * @param index
     *            The index of the parameter to get.
     * 
     * @return The Chow parameter at the given index.
     */
    BigInteger getChowParameter(int index) {
        if (!computedParameters[index]) {
            computeParameter(index);
        }
        return parameters[index];
    }

    /**
     * Gives the modified Chow parameter at the given index. If its value has
     * not been computed yet, invoking this method will compute it. Otherwise,
     * the cached value is returned.
     * 
     * @param index
     *            The index of the parameter to get.
     * 
     * @return The modified Chow parameter at the given index.
     */
    BigInteger getModifiedChowParameter(int index) {
        if (!computedParameters[index]) {
            computeParameter(index);
        }
        return modifiedParameters[index];
    }

    /**
     * Computes the value of the Chow parameter and modified Chow parameter for
     * the literal at the given index. The parameter is then marked as computed.
     * 
     * @param index
     *            The index of the parameter to compute.
     */
    private void computeParameter(int index) {
        BigInteger truePoints = parameters[size];

        if (!computedParameters[size]) {
            // The number of true points of the constraint has to be computed.
            truePoints = numberOfTruePoints();
            parameters[size] = truePoints;
            modifiedParameters[size] = truePoints
                    .subtract(BigInteger.ONE.shiftLeft(size - 1));
            computedParameters[size] = true;
        }

        if (index != size) {
            // Computing the value of the index-th parameter.
            parameters[index] = numberOfTruePoints(index);
            modifiedParameters[index] = parameters[index].shiftLeft(1)
                    .subtract(truePoints);
            computedParameters[index] = true;
        }
    }

    /**
     * Computes the number of true points of the original pseudo-Boolean
     * constraint.
     * 
     * @return The number of true points of the constraint.
     */
    BigInteger numberOfTruePoints() {
        return numberOfTruePoints(size, size, degree.intValue());
    }

    /**
     * Computes the number of true points of the original pseudo-Boolean
     * constraint in which a literal is assumed (i.e. supposed to be set to
     * true).
     * 
     * @param assumed
     *            The index of the assumed literal.
     * 
     * @return The number of true points of the constraint.
     */
    BigInteger numberOfTruePoints(int assumed) {
        return numberOfTruePoints(assumed, size - 1,
                degree.subtract(coefficients.get(assumed)).intValue());
    }

    /**
     * Computes the number of true points of the original pseudo-Boolean
     * constraint in which a literal is assumed (i.e. supposed to be set to
     * true). The given size and degree for the constraint are supposed to be
     * the result of updating the constraint after this assumption.
     * 
     * @param assumed
     *            The index of the assumed literal.
     * @param constraintSize
     *            The size of the constraint.
     * @param degree
     *            The degree of the constraint.
     * 
     * @return The number of true points of the constraint.
     */
    private BigInteger numberOfTruePoints(int assumed, int constraintSize,
            int degree) {
        if (degree <= 0) {
            // The constraint is valid.
            return BigInteger.ONE.shiftLeft(constraintSize);
        }

        // There is one counter-model falsifying all the literals.
        falsePoints[0][0] = BigInteger.ONE;

        // Considering the literals before the assumption.
        for (int i = 0; i < assumed; i++) {
            computeNumberOfFalsePoints(i + 1, coefficients.get(i), degree);
        }

        // Considering the literals after the assumption.
        for (int i = assumed + 1; i <= constraintSize; i++) {
            computeNumberOfFalsePoints(i, coefficients.get(i), degree);

        }

        // Deducing the number of models by counting the counter-models.
        BigInteger truePoints = BigInteger.ONE.shiftLeft(constraintSize);
        for (int s = 0; s < degree; s++) {
            truePoints = truePoints.subtract(falsePoints[constraintSize][s]);
        }
        return truePoints;
    }

    /**
     * Completes a row in {@link #falsePoints}.
     * 
     * @param index
     *            The index of the row to complete in {@link #falsePoints}.
     * @param coefficient
     *            The coefficient of the literal to consider.
     * @param degree
     *            The degree of the constraint.
     */
    private void computeNumberOfFalsePoints(int index, BigInteger coefficient,
            int degree) {
        int previous = index - 1;
        int saturated = Math.min(coefficient.intValue(), degree);

        falsePoints[index][0] = BigInteger.ONE;

        for (int s = 1; s < saturated; s++) {
            falsePoints[index][s] = falsePoints[previous][s];
        }

        for (int s = saturated; s < degree; s++) {
            falsePoints[index][s] = falsePoints[previous][s]
                    .add(falsePoints[previous][s - saturated]);
        }
    }

    public void setConstraint(IVecInt literals, IVec<BigInteger> coefficients,
            BigInteger degree) {
        this.literals = literals;
        this.coefficients = coefficients;
        this.degree = degree;
        this.size = literals.size();
        reset();
    }

}