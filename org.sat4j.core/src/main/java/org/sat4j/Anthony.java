package org.sat4j;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;

import org.sat4j.core.LiteralsUtils;
import org.sat4j.core.VecInt;
import org.sat4j.minisat.SolverFactory;
import org.sat4j.minisat.core.ICDCL;
import org.sat4j.minisat.core.ILits;
import org.sat4j.minisat.core.IOrder;
import org.sat4j.minisat.core.Solver;
import org.sat4j.reader.DimacsReader;
import org.sat4j.reader.ParseFormatException;
import org.sat4j.reader.Reader;
import org.sat4j.specs.ContradictionException;
import org.sat4j.specs.ISolver;
import org.sat4j.specs.IVecInt;
import org.sat4j.specs.IteratorInt;
import org.sat4j.specs.TimeoutException;
import org.sat4j.tools.Backbone;

public class Anthony {

    static class Decision {
        private final Integer d;
        private final IVecInt propagations;

        Decision(Integer d, IVecInt propagations) {
            this.d = d;
            this.propagations = propagations;
        }

        @Override
        public String toString() {
            return "(" + d + "|" + propagations + ")";
        }
    }

    private static IVecInt findPropagated(IVecInt implied,
            IVecInt assumptions) {
        IVecInt propagated = new VecInt();
        for (IteratorInt it = implied.iterator(); it.hasNext();) {
            int l = it.next();
            if (!assumptions.contains(l)) {
                propagated.push(l);
                assumptions.push(l);
            }
        }
        return propagated;
    }

    private static int decide(IVecInt implied, ILits lits, IOrder order) {
        for (IteratorInt it = implied.iterator(); it.hasNext();) {
            int l = it.next();
            lits.satisfies(LiteralsUtils.toInternal(l));
        }
        return LiteralsUtils.toDimacs(order.select());
    }

    public static void main(String[] args) throws TimeoutException,
            ParseFormatException, IOException, ContradictionException {
        if (args.length == 1) {
            String filename = args[0];
            ISolver solver = SolverFactory.newDefault();
            ILits lits = ((Solver) solver).getVocabulary();
            IOrder order = ((ICDCL<?>) solver).getOrder();

            Reader reader = new DimacsReader(solver);
            reader.parseInstance(filename);
            IVecInt implied = Backbone.instance().compute(solver);

            IVecInt assumptions = new VecInt();
            Collection<Decision> decisions = new ArrayList<>();
            IVecInt propagated = findPropagated(implied, assumptions);
            decisions.add(new Decision(null, propagated));
            do {
                int d = decide(implied, lits, order);
                assumptions.push(d);
                implied = Backbone.instance().compute(solver, assumptions);
                propagated = findPropagated(implied, assumptions);
                decisions.add(new Decision(d, propagated));
            } while (assumptions.size() < solver.realNumberOfVariables());
            System.out.println(decisions);
        }
    }

}
