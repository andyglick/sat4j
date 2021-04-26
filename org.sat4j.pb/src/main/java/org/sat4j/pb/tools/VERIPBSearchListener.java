package org.sat4j.pb.tools;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.math.BigInteger;

import org.sat4j.pb.constraints.pb.PBConstr;
import org.sat4j.specs.IConstr;
import org.sat4j.specs.IProblem;
import org.sat4j.specs.ISolverService;
import org.sat4j.specs.Lbool;
import org.sat4j.specs.RandomAccessModel;

public class VERIPBSearchListener implements PBSearchListener<ISolverService> {

    private static final long serialVersionUID = 1L;

    private StringBuilder conflict;
    private StringBuilder reason;
    private final String filename;
    private int nConstraints;

    public VERIPBSearchListener(String problemname) {
        this.filename = problemname.replace(".opb", ".pbp");
        File f = new File(filename);
        if (f.exists()) {
            f.delete();
        }
    }

    @Override
    public void init(ISolverService solverService) {
        this.nConstraints = ((IProblem) solverService).nConstraints();
        try {
            FileWriter fw = new FileWriter(this.filename, true);
            fw.write("pseudo-Boolean proof version 1.0\n");
            fw.write("f " + this.nConstraints + "\n");
            fw.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Override
    public void assuming(int p) {
    }

    @Override
    public void propagating(int p) {
    }

    @Override
    public void enqueueing(int p, IConstr reason) {
    }

    @Override
    public void backtracking(int p) {
    }

    @Override
    public void adding(int p) {
    }

    @Override
    public void learn(IConstr c) {
        try {
            FileWriter fw = new FileWriter(this.filename, true);
            fw.write("p " + this.conflict.toString() + "\n");
            fw.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        this.nConstraints++;
        c.setId(nConstraints);
    }

    @Override
    public void delete(IConstr c) {
        try {
            FileWriter fw = new FileWriter(this.filename, true);
            fw.write("d " + c.getId() + "\n");
            fw.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Override
    public void conflictFound(IConstr confl, int dlevel, int trailLevel) {
    }

    @Override
    public void conflictFound(int p) {
    }

    @Override
    public void solutionFound(int[] model, RandomAccessModel lazyModel) {
    }

    @Override
    public void beginLoop() {
    }

    @Override
    public void start() {
    }

    @Override
    public void end(Lbool result) {
        if (result.toString().equals("F")) {
            this.nConstraints++;
            try {
                FileWriter fw = new FileWriter(this.filename, true);
                fw.write("u >= 1 ;\n");
                fw.write("c " + this.nConstraints + "\n");
                fw.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    @Override
    public void restarting() {
    }

    @Override
    public void backjump(int backjumpLevel) {
    }

    @Override
    public void cleaning() {
    }

    @Override
    public void learnUnit(int p) {
        try {
            FileWriter fw = new FileWriter(this.filename, true);
            fw.write("p " + this.conflict.toString() + "\n");
            fw.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        this.nConstraints++;
    }

    @Override
    public void onConflict(PBConstr constr) {
        this.conflict = new StringBuilder("" + constr.getId());
    }

    @Override
    public void withReason(PBConstr constr) {
        this.reason = new StringBuilder("" + constr.getId());
    }

    @Override
    public void weakenOnReason(int p) {
        this.reason.append(" " + p).append(" w");
    }

    @Override
    public void weakenOnReason(BigInteger coeff, int p) {
        this.reason.append(" " + coeff).append(" " + p).append(" W");
    }

    @Override
    public void weakenOnConflict(int p) {
        this.conflict.append(" " + p).append(" w");
    }

    @Override
    public void weakenOnConflict(BigInteger coeff, int p) {
        this.conflict.append(" " + coeff).append(" " + p).append(" W");
    }

    @Override
    public void multiplyReason(BigInteger coeff) {
        this.reason.append(" " + coeff).append(" *");
    }

    @Override
    public void divideReason(BigInteger coeff) {
        this.reason.append(" " + coeff).append(" d");
    }

    @Override
    public void multiplyConflict(BigInteger coeff) {
        this.conflict.append(" " + coeff).append(" *");
    }

    @Override
    public void divideConflict(BigInteger coeff) {
        this.conflict.append(" " + coeff).append(" d");
    }

    @Override
    public void saturateReason() {
        this.reason.append(" s");
    }

    @Override
    public void saturateConflict() {
        this.conflict.append(" s");
    }

    @Override
    public void addReasonAndConflict() {
        this.conflict.append(" " + this.reason).append(" +");
    }

}