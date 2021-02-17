package org.sat4j.pb.constraints.pb;

import java.math.BigInteger;

public class ContinueAnalysisStrategy extends AbstractAnalysisStrategy {

    @Override
    public boolean shouldStopAfterAssertion(int currentLevel) {
        // We stop when the current level reaches the backtrack level.
        // This way, we do not need to restore decisions, as assignment are
        // removed at each resolution.
        // Said differently, instead of just canceling assignments, we (may)
        // have continued to perform resolutions for each undo.
        return currentLevel == assertiveDL;
    }

    @Override
    protected void resolveAfterAssertion(int pivotLit, ConflictMap conflict,
            PBConstr constr) {
        // A resolution is performed if and only if we can guarantee that the
        // backtrack level will not increase.
    }

    private void cuttingPlane(ConflictMap conflict, PBConstr constr,
            BigInteger degreeCons, BigInteger[] coefsCons, int indLitInConstr) {
        // cutting plane
        conflict.degree = conflict.cuttingPlane(constr, degreeCons, coefsCons,
                conflict.coefMultCons, solver, indLitInConstr);

        // saturation
        conflict.degree = conflict.saturation();
        conflict.divideCoefs();
    }

}
