/**
 * 
 */

package org.sat4j.tools;

import java.io.FileNotFoundException;
import java.io.OutputStream;
import java.io.PrintStream;

import org.sat4j.core.LiteralsUtils;
import org.sat4j.specs.IConstr;
import org.sat4j.specs.ISolverService;
import org.sat4j.specs.Lbool;
import org.sat4j.specs.SearchListenerAdapter;

/**
 * @author wallon
 *
 */
public class CdclTraceListener<S extends ISolverService>
        extends SearchListenerAdapter<S> {

    private transient S service;

    private final PrintStream output;

    public CdclTraceListener() {
        this(System.out);
    }

    public CdclTraceListener(String output) throws FileNotFoundException {
        this(new PrintStream(output));
    }

    public CdclTraceListener(OutputStream output) {
        this(new PrintStream(output));
    }

    public CdclTraceListener(PrintStream output) {
        this.output = output;
    }

    @Override
    public void init(S solverService) {
        this.service = solverService;
        output.printf("* #variable= %d #constraint= NBCONSTR%n",
                service.nVars());
    }

    @Override
    public void learn(IConstr c) {
        output.println(c.dump());
    }

    @Override
    public void learnUnit(int p) {
        int dimacs = LiteralsUtils.toDimacs(p);
        if (dimacs > 0) {
            output.printf("+1 x%d >= 1 ;%n", dimacs);

        } else {
            output.printf("+1 ~x%d >= 1 ;%n", -dimacs);
        }
    }

    @Override
    public void end(Lbool result) {
        if (output != System.out) {
            output.close();
        }
    }
}
