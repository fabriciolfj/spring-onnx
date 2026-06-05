package com.github.fabriciolfj.transactions.dtos;

import java.time.Instant;
import java.util.List;

public enum RiskLevel {
    LOW,
    MEDIUM,
    HIGH,
    CRITICAL;

    public static RiskLevel from(double probability) {
        if (probability >= 0.80) return LOW;
        if (probability >= 0.55) return MEDIUM;
        if (probability >= 0.30) return HIGH;
        return CRITICAL;
    }
}