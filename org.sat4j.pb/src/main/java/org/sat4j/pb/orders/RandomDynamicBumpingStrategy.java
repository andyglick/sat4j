package org.sat4j.pb.orders;

import java.util.Random;

import org.sat4j.OutputPrefix;
import org.sat4j.minisat.core.ConflictTimerAdapter;
import org.sat4j.minisat.core.DataStructureFactory;
import org.sat4j.minisat.core.ILits;
import org.sat4j.minisat.core.IOrder;
import org.sat4j.minisat.core.Solver;
import org.sat4j.pb.constraints.pb.PBConstr;

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
    private int index;

    public RandomDynamicBumpingStrategy(
            Solver<? extends DataStructureFactory> solver, int bound) {
        super(solver, bound);
        index = RANDOM.nextInt(bumpers.length);
        solver.addConflictTimer(this);
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
        index = RANDOM.nextInt(bumpers.length);
        // if (getSolver().isVerbose()) {
        System.out.println(OutputPrefix.COMMENT_PREFIX.toString()
                + " switching bumping strategy to " + bumpers[index]);
        // }
    }

    @Override
    public String toString() {
        return "Random bumping strategy apllied every " + bound()
                + " conflicts";
    }
}
