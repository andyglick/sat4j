package org.sat4j.pb;

/**
 * Solver Decorator to prevent the solver to receive a programmatic timeout
 * change.
 * 
 * It is expected to be useful when using {@link org.sat4j.tools.ManyCore} to
 * isolate the timeout of a particular solver compared to the general timeout
 * given to the other solvers.
 * 
 * @author leberre
 *
 */
public class PBTimeoutIsolator extends PBSolverDecorator {

    /**
     * 
     */
    private static final long serialVersionUID = 1L;

    public PBTimeoutIsolator(IPBSolver solver) {
        super(solver);
    }

    @Override
    public void setTimeoutOnConflicts(int count) {
        // ignore
    }

    @Override
    public void setTimeout(int t) {
        // ignore
    }

    @Override
    public void setTimeoutMs(long t) {
        // ignore
    }

}
