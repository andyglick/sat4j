package org.sat4j.pb.constraints.pb;

/**
 * The DefaultAnalysisStrategy implements the default strategy for stopping
 * conflict analysis, which does not perform any cancellation after the conflcit
 * has become assertive.
 * 
 * @author Romain Wallon
 */
public class DefaultAnalysisStrategy extends AbstractAnalysisStrategy {

    @Override
    public boolean shouldStopAfterAssertion(int currentLevel) {
        // The analysis stops immediately when the conflict becomes assertive.
        return true;
    }

    @Override
    protected void resolveAfterAssertion(int pivotLit, ConflictMap conflict,
            PBConstr constr) {
        throw new UnsupportedOperationException();
    }

}
