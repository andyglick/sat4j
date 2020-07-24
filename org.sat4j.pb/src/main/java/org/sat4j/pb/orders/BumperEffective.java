/**
 * 
 */
package org.sat4j.pb.orders;

import java.math.BigInteger;

import org.sat4j.core.VecInt;
import org.sat4j.minisat.core.ILits;
import org.sat4j.minisat.core.IOrder;
import org.sat4j.pb.constraints.pb.PBConstr;
import org.sat4j.specs.IVecInt;
import org.sat4j.specs.IteratorInt;

/**
 * 
 */
public class BumperEffective implements IBumper {

    private BigInteger constrDegree;

    private final IVecInt bumpableCandidates = new VecInt();

    private BumpStrategy bumpStrategy;

    @Override
    public void varBumpActivity(ILits voc, BumpStrategy bStrategy, IOrder order,
            PBConstr constr, int lit, int propagated) {
        if (lit == 0) {
            // A new constraint is being bumped.
            constrDegree = constr.getDegree();
            bumpableCandidates.clear();
            bumpStrategy = bStrategy;
        }

        if (lit == propagated) {
            // The implied literal is always bumped.
            bStrategy.varBumpActivity(order, constr, lit);

        } else if (!voc.isFalsified(constr.get(lit))) {
            // Weakening on this literal preserves the propagation.
            constrDegree = constrDegree.subtract(constr.getCoefs()[lit]);

        } else {
            // The literal may be effective.
            bumpableCandidates.push(lit);
        }
    }

    @Override
    public void postBumpActivity(IOrder order, PBConstr constr) {
        for (IteratorInt it = bumpableCandidates.iterator(); it.hasNext();) {
            int v = it.next();
            if (constr.getCoefs()[v].compareTo(constrDegree) >= 0) {
                bumpStrategy.varBumpActivity(order, constr, v);
            }
        }
    }

    @Override
    public String toString() {
        return "EFFECTIVE";
    }
}
