package org.sat4j.pb.tools;

import java.math.BigInteger;

import org.sat4j.pb.constraints.pb.PBConstr;
import org.sat4j.specs.ISolverService;
import org.sat4j.specs.SearchListenerAdapter;

public abstract class PBSearchListenerAdapter<S extends ISolverService>
        extends SearchListenerAdapter<S> implements PBSearchListener<S> {

    /**
     * 
     */
    private static final long serialVersionUID = 1L;

    @Override
    public void onConflict(PBConstr constr) {
    }

    @Override
    public void withReason(PBConstr constr) {
    }

    @Override
    public void weakenOnReason(int p) {
    }

    @Override
    public void weakenOnReason(BigInteger coeff, int p) {
    }

    @Override
    public void weakenOnConflict(int p) {
    }

    @Override
    public void weakenOnConflict(BigInteger coeff, int p) {
    }

    @Override
    public void multiplyReason(int coeff) {
    }

    @Override
    public void multiplyConflict(int coeff) {
    }

    @Override
    public void saturateReason() {
    }

    @Override
    public void saturateConflict() {
    }

    @Override
    public void addReasonAndConflict() {
    }

}
