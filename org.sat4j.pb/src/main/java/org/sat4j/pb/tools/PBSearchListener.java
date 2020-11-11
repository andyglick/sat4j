package org.sat4j.pb.tools;

import org.sat4j.annotations.Feature;
import org.sat4j.pb.constraints.pb.PBConstr;
import org.sat4j.specs.ISolverService;
import org.sat4j.specs.SearchListener;

/**
 * SearchListener for detailed Pseudo-Boolean solving.
 * 
 * @author leberre
 *
 * @param <S>
 * @since 2.3.6
 */
@Feature("searchlistener")
public interface PBSearchListener<S extends ISolverService>
        extends SearchListener<S> {

    /**
     * Called when a PB constraint is detected to be conflicting.
     * 
     * @param constr
     */
    void onConflict(PBConstr constr);

    /**
     * Called when considering a reason for conflict analysis.
     * 
     * @param constr
     */
    void withReason(PBConstr constr);

    /**
     * Weaken the current reason on a specific literal.
     * 
     * @param p
     *            a literal in dimacs format
     */
    void weakenOnReason(int p);

    /**
     * Weaken the current conflict on a specific literal.
     * 
     * @param p
     *            a literal in dimacs format
     */
    void weakenOnConflict(int p);

    /**
     * Multiply the current reason by an integer.
     * 
     * @param coeff
     *            the coefficient to apply of the reason
     */
    void multiplyReason(int coeff);

    /**
     * Multiply the current reason by an integer.
     * 
     * @param coeff
     *            the coefficient to apply of the reason
     */
    void multiplyConflict(int coeff);

    /**
     * Apply saturation on the reason side.
     */
    void saturateReason();

    /**
     * Apply saturation on the conflict side.
     */
    void saturateConflict();

    /**
     * Add the reason and the conflict.
     */
    void addReasonAndConflict();
}
