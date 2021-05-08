package org.sat4j.pb.orders;

import java.io.PrintStream;
import java.util.Random;

import org.sat4j.OutputPrefix;
import org.sat4j.minisat.core.ConflictTimerAdapter;
import org.sat4j.minisat.core.ILits;
import org.sat4j.minisat.core.IOrder;
import org.sat4j.pb.constraints.pb.PBConstr;
import org.sat4j.pb.core.PBSolverCP;

public class RandomDynamicBumpingStrategy extends ConflictTimerAdapter
        implements IBumper {

    /**
     * 
     */
    private static final long serialVersionUID = 1L;

    private final IBumper[] bumpers = new IBumper[] { Bumper.ANY,
            Bumper.ASSIGNED, Bumper.FALSIFIED, Bumper.FALSIFIED_AND_PROPAGATED,
            new BumperEffective(), new BumperEffectiveAndPropagated() };

    private static final Random RANDOM = new Random(12345);

    private final PrintStream out;
    private int index;

    private final PBSolverCP pbsolver;

    private long lastTimeMs = System.currentTimeMillis();
    private long lastDecision = 0L;

    public RandomDynamicBumpingStrategy(PBSolverCP solver, int bound) {
        super(solver, bound);
        index = RANDOM.nextInt(bumpers.length);
        solver.addConflictTimer(this);
        this.pbsolver = solver;
        out = System.err;
    }

    @Override
    public void varBumpActivity(ILits voc, BumpStrategy bumpStrategy,
            IOrder order, PBConstr constr, int i, int propagated,
            boolean conflicting) {
        bumpers[index].varBumpActivity(voc, bumpStrategy, order, constr, i,
                propagated, conflicting);
    }

    @Override
    public void postBumpActivity(IOrder order, PBConstr constr) {
        bumpers[index].postBumpActivity(order, constr);
    }

    @Override
    public void run() {
        out.println(buildStats());
        index = RANDOM.nextInt(bumpers.length);
        if (getSolver().isVerbose()) {
            System.out.println(OutputPrefix.COMMENT_PREFIX.toString()
                    + " switching bumping strategy to " + bumpers[index]);
        }
        int strategyIndex = RANDOM.nextInt(BumpStrategy.values().length);
        pbsolver.setBumpStrategy(BumpStrategy.values()[strategyIndex]);
        if (getSolver().isVerbose()) {
            System.out.println(OutputPrefix.COMMENT_PREFIX.toString()
                    + " switching bump strategy to "
                    + BumpStrategy.values()[strategyIndex]);
        }
    }

    private String buildStats() {
        StringBuilder stb = new StringBuilder();
        long deltatime = System.currentTimeMillis() - lastTimeMs;
        long deltadecision = pbsolver.getStats().getDecisions() - lastDecision;
        long depth = pbsolver.nAssigns();
        long dLevel = pbsolver.decisionLevel();
        stb.append("{\n");
        stb.append("  bumper: ");
        stb.append(bumpers[index]);
        stb.append(",\n");
        stb.append("  bumpStrategy: ");
        stb.append(pbsolver.getBumpStrategy());
        stb.append(",\n");
        stb.append("  time: ");
        stb.append(deltatime);
        stb.append(",\n");
        stb.append("  decisions: ");
        stb.append(deltadecision);
        stb.append(",\n");
        stb.append("  depth: ");
        stb.append(depth);
        stb.append(",\n");
        stb.append("  decisionLevel: ");
        stb.append(dLevel);
        stb.append("\n}");
        lastTimeMs = System.currentTimeMillis();
        lastDecision = pbsolver.getStats().getDecisions();
        return stb.toString();
    }

    @Override
    public String toString() {
        return "Random bumping strategy applied every " + bound()
                + " conflicts";
    }
}
