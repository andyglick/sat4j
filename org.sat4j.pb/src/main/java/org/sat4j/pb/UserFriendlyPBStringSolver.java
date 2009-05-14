/*******************************************************************************
 * SAT4J: a SATisfiability library for Java Copyright (C) 2004-2008 Daniel Le Berre
 *
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * which accompanies this distribution, and is available at
 * http://www.eclipse.org/legal/epl-v10.html
 *
 * Alternatively, the contents of this file may be used under the terms of
 * either the GNU Lesser General Public License Version 2.1 or later (the
 * "LGPL"), in which case the provisions of the LGPL are applicable instead
 * of those above. If you wish to allow use of your version of this file only
 * under the terms of the LGPL, and not to allow others to use your version of
 * this file under the terms of the EPL, indicate your decision by deleting
 * the provisions above and replace them with the notice and other provisions
 * required by the LGPL. If you do not delete the provisions above, a recipient
 * may use your version of this file under the terms of the EPL or the LGPL.
 * 
 * Based on the original MiniSat specification from:
 * 
 * An extensible SAT solver. Niklas Een and Niklas Sorensson. Proceedings of the
 * Sixth International Conference on Theory and Applications of Satisfiability
 * Testing, LNCS 2919, pp 502-518, 2003.
 *
 * See www.minisat.se for the original solver in C++.
 * 
 *******************************************************************************/
package org.sat4j.pb;

import java.math.BigInteger;
import java.util.Map;

import org.sat4j.specs.ContradictionException;
import org.sat4j.specs.IConstr;
import org.sat4j.specs.IVec;
import org.sat4j.specs.IVecInt;
import org.sat4j.specs.IteratorInt;
import org.sat4j.tools.DimacsStringSolver;

/**
 * Solver to display SAT instances using domain objects names instead of Dimacs
 * numbers.
 * 
 * @author leberre
 */
public class UserFriendlyPBStringSolver<T> extends DimacsStringSolver implements
		IPBSolver {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	private int indxConstrObj;

	private int nbOfConstraints;

	private ObjectiveFunction obj;

	private boolean inserted = false;

	private static IConstr FAKE_CONSTR = new IConstr() {

		public int size() {
			throw new UnsupportedOperationException("Fake IConstr");
		}

		public boolean learnt() {
			throw new UnsupportedOperationException("Fake IConstr");
		}

		public double getActivity() {
			throw new UnsupportedOperationException("Fake IConstr");
		}

		public int get(int i) {
			throw new UnsupportedOperationException("Fake IConstr");
		}
	};

	private Map<Integer, T> mapping;

	/**
	 * 
	 */
	public UserFriendlyPBStringSolver() {
	}

	/**
	 * @param initSize
	 */
	public UserFriendlyPBStringSolver(int initSize) {
		super(initSize);
	}

	public void setMapping(Map<Integer, T> mapping) {
		this.mapping = mapping;
	}

	public IConstr addPseudoBoolean(IVecInt lits, IVec<BigInteger> coeffs,
			boolean moreThan, BigInteger d) throws ContradictionException {
		StringBuffer out = getOut();
		assert lits.size() == coeffs.size();
		nbOfConstraints++;
		if (moreThan) {
			for (int i = 0; i < lits.size(); i++) {
				out.append(coeffs.get(i) + " " + mapping.get(lits.get(i))
						+ " + ");
			}
			out.append(">= " + d + " ;\n");
		} else {
			for (int i = 0; i < lits.size(); i++)
				out.append(coeffs.get(i) + " " + mapping.get(lits.get(i))
						+ " + ");
			out.append("<= " + d + " ;\n");
		}
		return FAKE_CONSTR;
	}

	public void setObjectiveFunction(ObjectiveFunction obj) {
		this.obj = obj;
	}

	@Override
	public IConstr addAtLeast(IVecInt literals, int degree)
			throws ContradictionException {
		StringBuffer out = getOut();
		nbOfConstraints++;
		for (IteratorInt iterator = literals.iterator(); iterator.hasNext();) {
			out.append(mapping.get(iterator.next()));
			out.append(" ");
			if (iterator.hasNext()) {
				out.append("+ ");
			}
		}
		out.append(">= " + degree + " ;\n");
		return FAKE_CONSTR;
	}

	@Override
	public IConstr addAtMost(IVecInt literals, int degree)
			throws ContradictionException {
		StringBuffer out = getOut();
		nbOfConstraints++;
		for (IteratorInt iterator = literals.iterator(); iterator.hasNext();) {
			out.append(mapping.get(iterator.next()));
			out.append(" ");
			if (iterator.hasNext()) {
				out.append("+ ");
			}
		}
		out.append("<= " + degree + " ;\n");
		return FAKE_CONSTR;
	}

	@Override
	public IConstr addClause(IVecInt literals) throws ContradictionException {
		StringBuffer out = getOut();
		nbOfConstraints++;
		int lit;
		boolean beginning = true;
		for (IteratorInt iterator = literals.iterator(); iterator.hasNext();) {
			lit = iterator.next();
			if (lit > 0) {
				if (beginning) {
					out.append("-> ");
					beginning = false;
				}
				out.append(mapping.get(lit));
			} else {
				out.append(mapping.get(-lit));
			}
			out.append(" ");
			if (iterator.hasNext() && !beginning) {
				out.append("OR ");
			}
		}
		out.append(" ;\n");
		return FAKE_CONSTR;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see org.sat4j.pb.IPBSolver#getExplanation()
	 */
	public String getExplanation() {
		// TODO Auto-generated method stub
		return null;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * org.sat4j.pb.IPBSolver#setListOfVariablesForExplanation(org.sat4j.specs
	 * .IVecInt)
	 */
	public void setListOfVariablesForExplanation(IVecInt listOfVariables) {
		// TODO Auto-generated method stub

	}

	@Override
	public String toString() {
		StringBuffer out = getOut();
		if (!inserted) {
			StringBuffer tmp = new StringBuffer();
			tmp.append("* #variable= " + nVars());
			tmp.append(" #constraint= " + nbOfConstraints + " \n");
			if (obj != null) {
				tmp.append("min: ");
				tmp.append(obj);
				tmp.append(" ;\n");
			}
			out.insert(indxConstrObj, tmp);
			inserted = true;
		}
		return out.toString();
	}

	@Override
	public String toString(String prefix) {
		return "OPB output solver";
	}

	@Override
	public int newVar(int howmany) {
		StringBuffer out = getOut();
		setNbVars(howmany);
		// to add later the number of constraints
		indxConstrObj = out.length();
		out.append("\n");
		return howmany;
	}

	@Override
	public void setExpectedNumberOfClauses(int nb) {
	}

	public ObjectiveFunction getObjectiveFunction() {
		return obj;
	}

}
