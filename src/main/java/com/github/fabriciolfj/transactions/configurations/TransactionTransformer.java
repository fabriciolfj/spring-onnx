package com.github.fabriciolfj.transactions.configurations;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.Shape;
import ai.djl.translate.NoBatchifyTranslator;
import ai.djl.translate.TranslatorContext;
import com.github.fabriciolfj.transactions.dtos.ApprovalResponse;
import com.github.fabriciolfj.transactions.dtos.TransactionRequest;

import java.util.List;
import java.util.stream.Stream;

public class TransactionTransformer implements NoBatchifyTranslator<TransactionRequest, ApprovalResponse> {

    private static final String REQUEST_KEY   = "request";
    private static final int    FEATURE_COUNT = 21;

    private final float threshold;
    private final float[] scalerMean;
    private final float[] scalerScale;
    private final FeatureDictionary.LabelEncoders encoders;

    public TransactionTransformer(final FeatureDictionary dict) {
        this.threshold   = dict.threshold();
        this.scalerMean  = dict.scaler().meanAsFloatArray();
        this.scalerScale = dict.scaler().scaleAsFloatArray();
        this.encoders    = dict.labelEncoders();
    }

    @Override
    public NDList processInput(final TranslatorContext ctx, final TransactionRequest request) {
        ctx.setAttachment(REQUEST_KEY, request);

        final float[] features = buildFeatureVector(request);
        final NDArray array    = ctx.getNDManager().create(features, new Shape(1, FEATURE_COUNT));

        return new NDList(array);
    }

    @Override
    public ApprovalResponse processOutput(final TranslatorContext ctx, final NDList list) {
        final float   probability = extractProbability(list);
        final var     request     = extractRequest(ctx);
        final var     reasons     = collectRiskReasons(request, probability);
        final boolean approved    = isApproved(request, probability);

        return approved
                ? ApprovalResponse.approved(probability, reasons)
                : ApprovalResponse.denied(probability, pickSuggestion(request), reasons);
    }
    private float[] buildFeatureVector(final TransactionRequest tx) {
        final float[] features = new float[FEATURE_COUNT];

        applyScaler(rawNumericFeatures(tx), features);
        applyCategoricalEncoding(tx, features);

        return features;
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

    // ── Regras de aprovação ───────────────────────────────────────────────────

    private boolean isApproved(final TransactionRequest tx, final float probability) {
        return !isHardBlocked(tx) && probability >= threshold;
    }

    private static boolean isHardBlocked(final TransactionRequest tx) {
        return exceedsDailyLimit(tx) || hasInsufficientBalance(tx);
    }

    private static boolean exceedsDailyLimit(final TransactionRequest tx) {
        return tx.amount().compareTo(tx.dailyLimit()) > 0;
    }

    private static boolean hasInsufficientBalance(final TransactionRequest tx) {
        return tx.amount().compareTo(tx.balance()) > 0;
    }

    private static boolean isFromForeignCountry(final TransactionRequest tx) {
        return !"BR".equals(tx.country());
    }

    private static boolean hasSuspiciousVelocity(final TransactionRequest tx) {
        return tx.secondsSinceLastTx() < 30;
    }

    private static boolean hasHighTxFrequency(final TransactionRequest tx) {
        return tx.txLast24h() > 15;
    }

    private static boolean hasLowCreditScore(final TransactionRequest tx) {
        return tx.creditScore() < 400;
    }

    private static boolean hasRepeatedFailedAttempts(final TransactionRequest tx) {
        return tx.failedAttempts24h() >= 3;
    }

    private static boolean isHighRiskScore(final float probability) {
        return probability < 0.30f;
    }

    private static List<String> collectRiskReasons(final TransactionRequest tx,
                                                   final float probability) {
        return Stream.of(
                        exceedsDailyLimit(tx)         ? "Valor acima do limite diário"                                         : null,
                        hasInsufficientBalance(tx)    ? "Saldo insuficiente"                                                   : null,
                        tx.isNewDevice()              ? "Dispositivo não reconhecido"                                          : null,
                        hasRepeatedFailedAttempts(tx) ? "Múltiplas tentativas falhas nas últimas 24h"                          : null,
                        isFromForeignCountry(tx)      ? "Transação originada em país estrangeiro (%s)".formatted(tx.country()) : null,
                        hasSuspiciousVelocity(tx)     ? "Intervalo muito curto desde a última transação (possível automação)"  : null,
                        hasLowCreditScore(tx)         ? "Score de crédito muito baixo"                                         : null,
                        hasHighTxFrequency(tx)        ? "Alta frequência de transações nas últimas 24h"                        : null,
                        isHighRiskScore(probability)  ? "Perfil de risco elevado detectado pelo modelo"                        : null
                )
                .filter(reason -> reason != null)
                .toList();
    }

    private static String pickSuggestion(final TransactionRequest tx) {
        if (exceedsDailyLimit(tx))         return "Valor excede o limite diário da conta";
        if (hasInsufficientBalance(tx))    return "Saldo insuficiente para realizar a transação";
        if (hasRepeatedFailedAttempts(tx)) return "Múltiplas tentativas falhas detectadas — conta em observação";
        if (tx.isNewDevice())              return "Transação de dispositivo não reconhecido — confirme sua identidade";
        if (isFromForeignCountry(tx))      return "Transação internacional — verifique se você iniciou esta operação";
        return                                    "Transação bloqueada por alto risco de fraude";
    }

    private static float extractProbability(final NDList list) {
        return list.getFirst().toFloatArray()[0];
    }

    private static TransactionRequest extractRequest(final TranslatorContext ctx) {
        return (TransactionRequest) ctx.getAttachment(REQUEST_KEY);
    }

    private static float toFloat(final boolean flag) {
        return flag ? 1.0f : 0.0f;
    }
}