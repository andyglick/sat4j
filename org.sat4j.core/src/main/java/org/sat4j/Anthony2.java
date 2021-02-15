package org.sat4j;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Iterator;
import java.util.stream.Collectors;

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

public class Anthony2 {

    static class Decision {
        private final Integer d;
        private final IVecInt propagations;

        Decision(Integer d, IVecInt propagations) {
            this.d = d;
            this.propagations = propagations;
        }

        Decision(Decision decisions) {
            this.d = decisions.d;
            this.propagations = decisions.propagations;
        }

        @Override
        public String toString() {
            return "(" + d + "|" + propagations + ")";
        }
    }

    static class Model {
        private final Collection<Decision> decisions;

        Model(Collection<Decision> decisions) {
            this.decisions = decisions;
        }

        @Override
        public String toString() {
            return this.decisions.toString();
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

    private static Collection<Decision> cloneDecisions(
            Collection<Decision> decisions) {
        return decisions.stream().map(Decision::new)
                .collect(Collectors.toList());
    }

    private static void applyAndDecide(Integer d,
            Collection<Decision> decisions, Collection<Model> models,
            IVecInt assumptions, ISolver solver, ILits lits, IOrder order)
            throws TimeoutException, ParseFormatException, IOException,
            ContradictionException {

        if (d != null) {
            assumptions.push(d);
        }
        System.out.println(assumptions);
        solver.clearLearntClauses();
        IVecInt implied = Backbone.instance().compute(solver, assumptions);
        IVecInt propagated = findPropagated(implied, assumptions);
        decisions.add(new Decision(d, propagated));
        if (assumptions.size() < solver.realNumberOfVariables()) {
            int nextDecision = decide(implied, lits, order);
            applyAndDecide(nextDecision, cloneDecisions(decisions), models,
                    assumptions.clone(), solver, lits, order);
            applyAndDecide(-nextDecision, cloneDecisions(decisions), models,
                    assumptions.clone(), solver, lits, order);
        } else {
            System.out.println(decisions);
            models.add(new Model(decisions));
        }
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

            IVecInt assumptions = new VecInt();
            Collection<Decision> decisions = new ArrayList<>();
            Collection<Model> models = new ArrayList<>();
            applyAndDecide(null, decisions, models, assumptions, solver, lits,
                    order);

            for (Iterator<Model> i = models.iterator(); i.hasNext();) {
                System.out.println(i.next());
            }
        }
    }

}
