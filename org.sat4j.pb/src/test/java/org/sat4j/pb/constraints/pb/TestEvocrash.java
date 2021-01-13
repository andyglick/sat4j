package org.sat4j.pb.constraints.pb;

import static org.junit.Assert.fail;

import org.junit.Test;
import org.sat4j.core.VecInt;
import org.sat4j.minisat.constraints.cnf.Lits;

public class TestEvocrash {

    @Test(expected = ArrayIndexOutOfBoundsException.class, timeout = 4000)
    public void test0() throws Throwable {
        Lits lits0 = new Lits();
        IDataStructurePB iDataStructurePB0 = IDataStructurePB.TAUTOLOGY;
        WatchPbLongCP watchPbLongCP0 = MinWatchPbLongCP
                .normalizedWatchPbNew(lits0, iDataStructurePB0);
        ConflictMapClause conflictMapClause0 = new ConflictMapClause(
                watchPbLongCP0, 0, false);
        VecInt[] vecIntArray0 = new VecInt[9];
        VecInt vecInt0 = new VecInt(1, 1);
        vecIntArray0[1] = vecInt0;
        conflictMapClause0.byLevel = vecIntArray0;
        // Undeclared exception!
        conflictMapClause0.isAssertive(0);
        fail("Expecting exception: ArrayIndexOutOfBoundsException");

    }
}
