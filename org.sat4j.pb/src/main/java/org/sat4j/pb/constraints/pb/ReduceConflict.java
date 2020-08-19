/**
 * 
 */
package org.sat4j.pb.constraints.pb;

import java.math.BigInteger;

import org.sat4j.core.VecInt;
import org.sat4j.specs.IVecInt;
import org.sat4j.specs.IteratorInt;

/**
 * @author romainwallon
 *
 */
public class ReduceConflict implements IReduceConflictStrategy {

    private static final IReduceConflictStrategy INSTANCE = new ReduceConflict();

    private ReduceConflict() {
        // TODO Auto-generated constructor stub
    }

    public static final IReduceConflictStrategy instance() {
        return INSTANCE;
    }

    @Override
    public boolean reduceConflict(ConflictMapDivideByPivot conflict,
            int literal) {
        BigInteger coef = conflict.weightedLits.get(literal);
        int size = conflict.weightedLits.size();
        BigInteger[] coefficients = new BigInteger[size];
        int[] literals = new int[size];
        BigInteger degree = conflict.degree;
        int indLit = -1;

        // Weakening away the literals that are not divisible.
        for (int i = 0; i < size; i++) {
            coefficients[i] = conflict.weightedLits.getCoef(i);
            literals[i] = conflict.weightedLits.getLit(i);
            if (literals[i] == literal) {
                indLit = i;
            }
            if (!conflict.voc.isFalsified(conflict.weightedLits.getLit(i))) {
                if (coefficients[i].mod(coef).signum() != 0) {
                    degree = degree.subtract(coefficients[i]);
                    coefficients[i] = BigInteger.ZERO;
                }
            }
        }

        degree = conflict.irrelevantLiteralRemover.remove(conflict.voc.nVars(),
                literals, coefficients, degree, conflict.stats, true);

        if (coefficients[indLit].signum() == 0) {
            return false;
        }

        IVecInt toRemove = new VecInt();
        for (int i = 0; i < literals.length; i++) {
            int l = literals[i];
            if (coefficients[i].signum() == 0) {
                toRemove.push(l);
            } else {
                conflict.setCoef(l, ConflictMapDivideByPivot
                        .ceildiv(coefficients[i], coef));
            }
        }

        for (IteratorInt it = toRemove.iterator(); it.hasNext();) {
            conflict.removeCoef(it.next());
        }

        conflict.degree = ConflictMapDivideByPivot.ceildiv(degree, coef);
        conflict.saturation();
        conflict.coefMultCons = BigInteger.ONE;
        conflict.stats.incNumberOfRoundingOperations();
        return true;
    }

}
