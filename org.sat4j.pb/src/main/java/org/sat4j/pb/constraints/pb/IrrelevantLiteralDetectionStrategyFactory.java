package org.sat4j.pb.constraints.pb;

public final class IrrelevantLiteralDetectionStrategyFactory {

    private IrrelevantLiteralDetectionStrategyFactory() {
        // Disables instantiation.
    }

    public static IrrelevantLiteralDetectionStrategy defaultStrategy() {
        return ChineseRemaindersIrrelevantLiteralDetectionStrategy
                .forPrimes(401, 307, 199, 101);
    }

}
