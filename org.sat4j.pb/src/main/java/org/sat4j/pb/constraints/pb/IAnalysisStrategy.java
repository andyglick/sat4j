/**
 * 
 */
package org.sat4j.pb.constraints.pb;

import org.sat4j.pb.core.PBSolverCP;

/**
 * @author romainwallon
 *
 */
public interface IAnalysisStrategy {

    void setSolver(PBSolverCP solver);

    void isAssertiveAt(int dl, int litImplied);

    void resolve(int pivotLit, ConflictMap conflict, PBConstr constr);

    boolean shouldStop(int currentLevel);

}
