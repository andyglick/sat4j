/**
 * 
 */
package org.sat4j.pb.constraints.pb;

import org.sat4j.pb.core.PBSolverCP;
import org.sat4j.pb.core.PBSolverStats;

/**
 * The AbstractAnalysisStrategy defines template methods for implementing
 * {@link IAnalysisStrategy}.
 * 
 * @author Romain Wallon
 */
public abstract class AbstractAnalysisStrategy implements IAnalysisStrategy {

    /**
     * The solver which uses this strategy.
     */
    protected PBSolverCP solver;

    /**
     * The decision level at which the conflict is assertive.
     */
    protected int assertiveDL = -2;

    /**
     * The index of the assertive literal.
     */
    protected int assertiveLit = -1;

    /**
     * Whether the cancellation to perform after the assertion is the first.
     */
    protected boolean firstCancellationAfterAssertion = true;

    @Override
    public void setSolver(PBSolverCP solver) {
        this.solver = solver;
    }

    @Override
    public void newConflict() {
        this.assertiveDL = -2;
        this.assertiveLit = -1;
        this.firstCancellationAfterAssertion = true;
    }

    @Override
    public void isAssertiveAt(int dl, int assertiveLit) {
        if (dl < assertiveDL) {
            ((PBSolverStats) solver.getStats()).incNbImprovedBackjumps();
        }
        this.assertiveDL = dl;
        this.assertiveLit = assertiveLit;
    }

    /**
     * Checks whether the conflict being analyzed has already been assertive.
     *
     * @return Whether the conflict being analyzed has already been assertive.
     */
    protected boolean hasBeenAssertive() {
        return assertiveDL != -2;
    }

    @Override
    public void resolve(int pivotLit, ConflictMap conflict, PBConstr constr) {
        if (hasBeenAssertive()) {
            // A specific kind of resolution should be applied to preserve
            // assertion.
            resolveAfterAssertion(pivotLit, conflict, constr);

        } else {
            // Resolution is performed as usual.
            conflict.resolve(constr, pivotLit, solver);
        }
    }

    /**
     * Cancels out a literal by resolving two constraints, while the conflict
     * being analyzed is already assertive.
     * 
     * @param pivotLit
     *            The literal to cancel.
     * @param conflict
     *            The conflict being analyzed.
     * @param constr
     *            The constraint to resolve with the conflict.
     */
    protected abstract void resolveAfterAssertion(int pivotLit,
            ConflictMap conflict, PBConstr constr);

    @Override
    public boolean shouldStop(int currentLevel) {
        return hasBeenAssertive() && shouldStopAfterAssertion(currentLevel);
    }

    /**
     * Checks whether the current analysis should stop, while the conflict has
     * already become assertive.
     * 
     * @param currentLevel
     *            The last decision level on the trail.
     * 
     * @return Whether the analysis should stop.
     */
    protected abstract boolean shouldStopAfterAssertion(int currentLevel);

}
