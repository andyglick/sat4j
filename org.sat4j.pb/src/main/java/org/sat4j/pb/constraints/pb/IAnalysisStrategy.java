/**
 * 
 */

package org.sat4j.pb.constraints.pb;

import org.sat4j.pb.core.PBSolverCP;

/**
 * The IAnalysisStrategy defines the interface for the detecting when to stop
 * the conflict analysis procedure.
 * 
 * @author Romain Wallon
 */
public interface IAnalysisStrategy {

    /**
     * Sets the solver that uses this strategy.
     * 
     * @param solver
     *            The solver that uses this strategy.
     */
    void setSolver(PBSolverCP solver);

    /**
     * Notifies this strategy that a new conflict is being analyzed.
     */
    void newConflict();

    /**
     * Notifies this strategy that the current conflict is assertive at the
     * given decision level.
     * 
     * @param dl
     *            The decision level at which the conflict is assertive.
     * @param litImplied
     *            The literal that is implied by the assertive constraint.
     */
    void isAssertiveAt(int dl, int litImplied);

    /**
     * Cancels out a literal by resolving two constraints.
     * 
     * @param pivotLit
     *            The literal to cancel.
     * @param conflict
     *            The conflict being analyzed.
     * @param constr
     *            The constraint to resolve with the conflict.
     */
    void resolve(int pivotLit, ConflictMap conflict, PBConstr constr);

    /**
     * Checks whether the current analysis should stop.
     * 
     * @param currentLevel
     *            The last decision level on the trail.
     * 
     * @return Whether the analysis should stop.
     */
    boolean shouldStop(int currentLevel);

}
