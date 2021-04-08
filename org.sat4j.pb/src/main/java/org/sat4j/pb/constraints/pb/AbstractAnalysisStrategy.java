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

    protected int assertiveLitIndex;

    @Override
    public void setSolver(PBSolverCP solver) {
        this.solver = solver;
    }

    @Override
    public void reset() {
        this.assertiveDL = -2;
    }

    @Override
    public void isAssertiveAt(int dl, int assertiveLit) {
        this.assertiveDL = dl;
        this.assertiveLitIndex = assertiveLit;
    }

    @Override
    public void resolve(int pivotLit, ConflictMap conflict, PBConstr constr) {
        if (hasBeenAssertive()) {
            resolveAfterAssertion(pivotLit, conflict, constr);
        } else {
            conflict.resolve(constr, pivotLit, solver);
        }
    }

    protected abstract void resolveAfterAssertion(int pivotLit,
            ConflictMap conflict, PBConstr constr);

    protected boolean hasBeenAssertive() {
        return assertiveDL != -2;
    }

    @Override
    public boolean shouldStop(int currentLevel) {
        return hasBeenAssertive() && shouldStopAfterAssertion(currentLevel);
    }

    protected abstract boolean shouldStopAfterAssertion(int currentLevel);
}
