package org.sat4j.pb.constraints.pb;

public class DefaultAnalysisStrategy extends AbstractAnalysisStrategy {

    @Override
    public boolean shouldStopAfterAssertion(int currentLevel) {
        return true;
    }

    @Override
    protected void resolveAfterAssertion(int pivotLit, ConflictMap conflict,
            PBConstr constr) {
        throw new UnsupportedOperationException();
    }

}
