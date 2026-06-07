package com.github.fabriciolfj.transactions.configurations;

import com.github.fabriciolfj.transactions.dtos.TransactionRequest;

import static com.github.fabriciolfj.transactions.configurations.TransactionTransformer.FEATURE_COUNT;

public class MapperInput {

    private final float[] scalerMean;
    private final float[] scalerScale;
    private final FeatureDictionary.LabelEncoders encoders;


    public MapperInput(float[] scalerMean, float[] scalerScale, FeatureDictionary.LabelEncoders encoders) {
        this.scalerMean = scalerMean;
        this.scalerScale = scalerScale;
        this.encoders = encoders;
    }

    public float[] buildFeatureVector(final TransactionRequest tx) {
        final float[] features = new float[FEATURE_COUNT];

        applyScaler(rawNumericFeatures(tx), features);
        applyCategoricalEncoding(tx, features);

        return features;
    }

    /** z = (x − mean) / scale para cada feature numérica. */
    private void applyScaler(final float[] raw, final float[] out) {
        for (int i = 0; i < scalerMean.length; i++) {
            out[i] = (raw[i] - scalerMean[i]) / scalerScale[i];
        }
    }

    private void applyCategoricalEncoding(final TransactionRequest tx, final float[] out) {
        out[17] = encoders.transactionType().encode(tx.transactionType().name());
        out[18] = encoders.merchantCategory().encode(tx.merchantCategory().name());
        out[19] = encoders.country().encode(tx.country());
        out[20] = encoders.accountTier().encode(tx.accountTier().name());
    }

    private static float[] rawNumericFeatures(final TransactionRequest tx) {
        return new float[] {
                tx.amount().floatValue(),           // [0]
                tx.hour(),                          // [1]
                tx.dayOfWeek(),                     // [2]
                toFloat(tx.isWeekend()),            // [3]
                tx.accountAgeDays(),                // [4]
                tx.balance().floatValue(),          // [5]
                tx.dailyLimit().floatValue(),       // [6]
                tx.monthlyLimit().floatValue(),     // [7]
                tx.creditScore(),                   // [8]
                tx.txLast24h(),                     // [9]
                tx.txLast7d(),                      // [10]
                tx.avgTxAmount30d().floatValue(),   // [11]
                tx.secondsSinceLastTx(),            // [12]
                toFloat(tx.isNewDevice()),          // [13]
                toFloat(tx.isNewBeneficiary()),     // [14]
                tx.failedAttempts24h(),             // [15]
                tx.deviceTrustScore().floatValue()  // [16]
        };
    }

    private static float toFloat(final boolean flag) {
        return flag ? 1.0f : 0.0f;
    }
}
