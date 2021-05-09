package org.sat4j.pb.orders;

import java.io.IOException;
import java.io.PrintStream;
import java.net.Socket;
import java.util.Map;
import java.util.Scanner;

import org.sat4j.OutputPrefix;
import org.sat4j.minisat.core.ConflictTimerAdapter;
import org.sat4j.minisat.core.ILits;
import org.sat4j.minisat.core.IOrder;
import org.sat4j.pb.constraints.pb.PBConstr;
import org.sat4j.pb.core.PBSolverCP;

public class ExternalDynamicBumpingStrategy extends ConflictTimerAdapter
        implements IBumper {

    /**
     * 
     */
    private static final long serialVersionUID = 1L;

    private final Map<String, IBumper> bumpers = Map.of("ANY", Bumper.ANY,
            "ASSIGNED", Bumper.ASSIGNED, "FALSIFIED", Bumper.FALSIFIED,
            "FALSIFIED_AND_PROPAGATED", Bumper.FALSIFIED_AND_PROPAGATED,
            "EFFECTIVE", new BumperEffective(), "EFFECTIVE_AND_PROPAGATED",
            new BumperEffectiveAndPropagated());

    private PrintStream out;

    private Scanner in;
    private String bumper = "ANY";

    private final PBSolverCP pbsolver;

    private long lastTimeMs = System.currentTimeMillis();
    private long lastDecision = 0L;

    public ExternalDynamicBumpingStrategy(PBSolverCP solver, int bound,
            int port) {
        super(solver, bound);
        solver.addConflictTimer(this);
        this.pbsolver = solver;

        try {
            Socket socket = new Socket("127.0.0.1", port);
            out = new PrintStream(socket.getOutputStream());
            in = new Scanner(socket.getInputStream());
        } catch (IOException e) {
            out = System.err;
            in = new Scanner(System.in);
        }
    }

    @Override
    public void varBumpActivity(ILits voc, BumpStrategy bumpStrategy,
            IOrder order, PBConstr constr, int i, int propagated,
            boolean conflicting) {
        bumpers.get(bumper).varBumpActivity(voc, bumpStrategy, order, constr, i,
                propagated, conflicting);
    }

    @Override
    public void postBumpActivity(IOrder order, PBConstr constr) {
        bumpers.get(bumper).postBumpActivity(order, constr);
    }

    @Override
    public void run() {
        out.println(buildStats());
        String jsonline = in.nextLine();
        String[] parts = jsonline.split(" ");
        if (parts.length != 4) {
            throw new IllegalStateException(
                    "Wrong format from DAC engine: " + jsonline);
        }
        bumper = parts[1].substring(1, parts[1].length() - 2);
        String bumpStrategy = parts[3].substring(1, parts[3].length() - 2);

        if (getSolver().isVerbose()) {
            System.out.println(OutputPrefix.COMMENT_PREFIX.toString()
                    + " switching bumping strategy to " + bumper);
        }
        pbsolver.setBumpStrategy(BumpStrategy.valueOf(bumpStrategy));
        if (getSolver().isVerbose()) {
            System.out.println(OutputPrefix.COMMENT_PREFIX.toString()
                    + " switching bump strategy to " + bumpStrategy);
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
        stb.append(bumper);
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
