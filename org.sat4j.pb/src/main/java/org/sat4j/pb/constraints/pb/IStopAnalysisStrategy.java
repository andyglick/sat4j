package org.sat4j.pb.constraints.pb;

@FunctionalInterface
public interface IStopAnalysisStrategy {

    boolean shouldStop(int assertiveLevel, int currentLevel);

}
