package com.github.fabriciolfj.transactions.configurations;

import ai.djl.ndarray.NDList;
import ai.djl.translate.TranslatorContext;
import com.github.fabriciolfj.transactions.dtos.TransactionRequest;

import java.util.List;
import java.util.Objects;
import java.util.stream.Stream;

import static com.github.fabriciolfj.transactions.configurations.TransactionTransformer.REQUEST_KEY;

public class MapperOutput {

    private MapperOutput() { }

    public static float extractProbability(final NDList list) {
        return list.getFirst().toFloatArray()[0];
    }

    public static List<String> collectRiskReasons(final TransactionRequest tx,
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
                .filter(Objects::nonNull)
                .toList();
    }

    // ── Regras de aprovação ───────────────────────────────────────────────────

    public static boolean isApproved(final TransactionRequest tx, final float probability, final float threshold) {
        return !isHardBlocked(tx) && probability >= threshold;
    }


    public static String pickSuggestion(final TransactionRequest tx) {
        if (exceedsDailyLimit(tx))         return "Valor excede o limite diário da conta";
        if (hasInsufficientBalance(tx))    return "Saldo insuficiente para realizar a transação";
        if (hasRepeatedFailedAttempts(tx)) return "Múltiplas tentativas falhas detectadas — conta em observação";
        if (tx.isNewDevice())              return "Transação de dispositivo não reconhecido — confirme sua identidade";
        if (isFromForeignCountry(tx))      return "Transação internacional — verifique se você iniciou esta operação";
        return                                    "Transação bloqueada por alto risco de fraude";
    }

    public static TransactionRequest extractRequest(final TranslatorContext ctx) {
        return (TransactionRequest) ctx.getAttachment(REQUEST_KEY);
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
}
