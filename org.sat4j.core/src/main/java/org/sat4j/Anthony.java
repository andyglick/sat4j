package org.sat4j;

import java.io.IOException;

import org.sat4j.minisat.SolverFactory;
import org.sat4j.reader.DimacsReader;
import org.sat4j.reader.ParseFormatException;
import org.sat4j.reader.Reader;
import org.sat4j.specs.ContradictionException;
import org.sat4j.specs.ISolver;
import org.sat4j.specs.TimeoutException;
import org.sat4j.tools.Backbone;

public class Anthony {
    public static void main(String[] args) throws TimeoutException,
            ParseFormatException, IOException, ContradictionException {
        if (args.length == 1) {
            String filename = args[0];
            ISolver solver = SolverFactory.newDefault();
            Reader reader = new DimacsReader(solver);
            reader.parseInstance(filename);
            System.out.println(Backbone.instance().compute(solver));
        }
    }
}
