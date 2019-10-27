package org.sat4j.pb;

import java.io.PrintWriter;
import java.math.BigInteger;
import java.util.LinkedList;
import java.util.List;
import java.util.Map.Entry;
import java.util.NavigableMap;
import java.util.TreeMap;

import org.sat4j.core.Vec;
import org.sat4j.pb.constraints.pb.IrrelevantLiteralDetectionStrategy;
import org.sat4j.specs.ContradictionException;
import org.sat4j.specs.FakeConstr;
import org.sat4j.specs.IConstr;
import org.sat4j.specs.IVec;
import org.sat4j.specs.IVecInt;
import org.sat4j.specs.IteratorInt;
import org.sat4j.tools.AbstractOutputSolver;

public class DetectIrrelevantSolver extends AbstractOutputSolver
        implements IPBSolver {

    private int nbIrrelevant = 0;

    private int nVars = 0;

    private int nConstraints = 0;

    private final IrrelevantLiteralDetectionStrategy irrelevantDetector = IrrelevantLiteralDetectionStrategy
            .defaultStrategy();

    @Override
    public int newVar() {
        return newVar(1);
    }

    @Override
    public int newVar(int howmany) {
        nVars += howmany;
        return nVars;
    }

    @Override
    public int nextFreeVarId(boolean reserve) {
        return 0;
    }

    @Override
    public void registerLiteral(int p) {
    }

    @Override
    public void setExpectedNumberOfClauses(int nb) {
    }

    @Override
    public IConstr addClause(IVecInt literals) throws ContradictionException {
        // All literals are relevant.
        nConstraints++;
        return FakeConstr.instance();
    }

    @Override
    public IConstr addAtMost(IVecInt literals, int degree)
            throws ContradictionException {
        // All literals are relevant.
        nConstraints++;
        return FakeConstr.instance();
    }

    @Override
    public IConstr addAtLeast(IVecInt literals, int degree)
            throws ContradictionException {
        // All literals are relevant.
        nConstraints++;
        return FakeConstr.instance();
    }

    @Override
    public IConstr addExactly(IVecInt literals, int n)
            throws ContradictionException {
        // All literals are relevant.
        nConstraints++;
        return FakeConstr.instance();
    }

    @Override
    public IConstr addParity(IVecInt literals, boolean even) {
        // All literals are relevant.
        nConstraints++;
        return FakeConstr.instance();
    }

    @Override
    public void reset() {
        nbIrrelevant = 0;
    }

    @Override
    public void printStat(PrintWriter out) {
        out.println("c number of variables:\t" + nVars);
        out.println("c number of constraints:\t" + nConstraints);
        out.println("c number of irrelevant literals:\t" + nbIrrelevant);
    }

    @Override
    public String toString(String prefix) {
        return prefix + "Sat4j counter for irrelevant literals.";
    }

    @Override
    public int[] modelWithInternalVariables() {
        return new int[0];
    }

    @Override
    public int realNumberOfVariables() {
        return 0;
    }

    @Override
    public boolean primeImplicant(int p) {
        return false;
    }

    @Override
    public void printInfos(PrintWriter out) {
    }

    @Override
    public IConstr addPseudoBoolean(IVecInt lits, IVec<BigInteger> coeffs,
            boolean moreThan, BigInteger d) throws ContradictionException {
        if (moreThan) {
            return addAtLeast(lits, coeffs, d);
        }
        return addAtMost(lits, coeffs, d);
    }

    @Override
    public IConstr addAtMost(IVecInt literals, IVecInt coeffs, int degree)
            throws ContradictionException {
        Vec<BigInteger> bigCoeffs = new Vec<>(coeffs.size());
        for (IteratorInt it = coeffs.iterator(); it.hasNext();) {
            bigCoeffs.push(BigInteger.valueOf(it.next()));
        }
        return addAtMost(literals, bigCoeffs, BigInteger.valueOf(degree));
    }

    @Override
    public IConstr addAtMost(IVecInt literals, IVec<BigInteger> coeffs,
            BigInteger degree) throws ContradictionException {
        BigInteger sumAllCoeffs = BigInteger.ZERO;
        for (int i = 0; i < coeffs.size(); i++) {
            literals.set(i, -literals.get(i));
            sumAllCoeffs = sumAllCoeffs.add(coeffs.get(i));
        }
        return addAtLeast(literals, coeffs, sumAllCoeffs.subtract(degree));
    }

    @Override
    public IConstr addAtLeast(IVecInt literals, IVecInt coeffs, int degree)
            throws ContradictionException {
        Vec<BigInteger> bigCoeffs = new Vec<>(coeffs.size());
        for (IteratorInt it = coeffs.iterator(); it.hasNext();) {
            bigCoeffs.push(BigInteger.valueOf(it.next()));
        }
        return addAtLeast(literals, bigCoeffs, BigInteger.valueOf(degree));
    }

    @Override
    public IConstr addAtLeast(IVecInt literals, IVec<BigInteger> coeffs,
            BigInteger degree) throws ContradictionException {
        nConstraints++;
        if (literals.size() >= 500 || literals.size() == 1) {
            return FakeConstr.instance();
        }
        NavigableMap<BigInteger, List<Integer>> sortedCoeffs = new TreeMap<>();
        BigInteger maxCoeff = BigInteger.ZERO;
        BigInteger realDegree = degree;
        for (int i = 0; i < literals.size(); i++) {
            BigInteger c = coeffs.get(i);
            if (c.signum() < 0) {
                c = c.negate();
                coeffs.set(i, c);
                literals.set(i, -literals.get(i));
                realDegree = realDegree.add(c);
            }
            sortedCoeffs.computeIfAbsent(c, k -> new LinkedList<Integer>())
                    .add(i);
            maxCoeff = maxCoeff.max(c);
        }
        int irr = 0;
        for (Entry<BigInteger, List<Integer>> e : sortedCoeffs.entrySet()) {
            if (dependsOn(literals, coeffs, realDegree, e.getKey(),
                    e.getValue().get(0))) {
                break;
            }
            irr += e.getValue().size();
        }
        nbIrrelevant += irr;
        return FakeConstr.instance();

    }

    private boolean dependsOn(IVecInt literals, IVec<BigInteger> coeffs,
            BigInteger degree, BigInteger coef, int lit) {
        return irrelevantDetector.dependsOn(nVars, literals, coeffs, degree,
                lit, coef);
    }

    @Override
    public IConstr addExactly(IVecInt literals, IVecInt coeffs, int weight)
            throws ContradictionException {
        addAtMost(literals, coeffs, weight);
        addAtLeast(literals, coeffs, weight);
        return FakeConstr.instance();
    }

    @Override
    public IConstr addExactly(IVecInt literals, IVec<BigInteger> coeffs,
            BigInteger weight) throws ContradictionException {
        addAtMost(literals, coeffs, weight);
        addAtLeast(literals, coeffs, weight);
        return FakeConstr.instance();
    }

    private ObjectiveFunction obj;

    @Override
    public void setObjectiveFunction(ObjectiveFunction obj) {
        this.obj = obj;
    }

    @Override
    public ObjectiveFunction getObjectiveFunction() {
        return obj;
    }

}
