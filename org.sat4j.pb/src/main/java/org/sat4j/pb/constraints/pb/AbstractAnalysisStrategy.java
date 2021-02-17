/**
 * 
 */
package org.sat4j.pb.constraints.pb;

import org.sat4j.pb.core.PBSolverCP;

/**
 * @author romainwallon
 *
 */
public abstract class AbstractAnalysisStrategy implements IAnalysisStrategy {

    protected PBSolverCP solver;

    protected int assertiveDL = -2;

    @Override
    public void setSolver(PBSolverCP solver) {
        this.solver = solver;
    }

    @Override
    public void isAssertiveAt(int dl) {
        this.assertiveDL = dl;
    }

    @Override
    public void resolve(int pivotLit, ConflictMap conflict, PBConstr constr) {
        if (!hasBeenAssertive()) {
            conflict.resolve(constr, pivotLit, solver);
        } else {
            resolveAfterAssertion(pivotLit, conflict, constr);
        }
    }

    protected abstract void resolveAfterAssertion(int pivotLit,
            ConflictMap conflict, PBConstr constr);

    protected boolean hasBeenAssertive() {
        return assertiveDL == -2;
    }

    @Override
    public boolean shouldStop(int currentLevel) {
        return hasBeenAssertive() && shouldStopAfterAssertion(currentLevel);
    }

    protected abstract boolean shouldStopAfterAssertion(int currentLevel);
}
