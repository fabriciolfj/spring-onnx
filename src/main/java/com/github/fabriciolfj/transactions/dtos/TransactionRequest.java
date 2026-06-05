package com.github.fabriciolfj.transactions.dtos;

import com.fasterxml.jackson.annotation.JsonProperty;
import jakarta.validation.constraints.*;
import java.math.BigDecimal;

public record TransactionRequest(
        @NotNull
        @DecimalMin(value = "0.01", message = "O valor mínimo da transação é R$ 0,01")
        @Digits(integer = 12, fraction = 2)
        @JsonProperty("amount")
        BigDecimal amount,

        @NotNull
        @JsonProperty("transactionType")
        TransactionType transactionType,

        @NotNull
        @JsonProperty("merchantCategory")
        MerchantCategory merchantCategory,

        @NotBlank
        @Size(min = 2, max = 2)
        @JsonProperty("country")
        String country,

        @NotNull
        @Min(0)
        @Max(23)
        @JsonProperty("hour")
        Integer hour,

        @NotNull
        @Min(0) @Max(6)
        @JsonProperty("dayOfWeek")
        Integer dayOfWeek,

        @NotNull
        @JsonProperty("isWeekend")
        Boolean isWeekend,

        @NotNull
        @JsonProperty("accountTier")
        AccountTier accountTier,

        @NotNull
        @Min(0)
        @JsonProperty("accountAgeDays")
        Integer accountAgeDays,

        @NotNull
        @DecimalMin("0.00")
        @Digits(integer = 15, fraction = 2)
        @JsonProperty("balance")
        BigDecimal balance,

        @NotNull
        @DecimalMin("0.00")
        @Digits(integer = 12, fraction = 2)
        @JsonProperty("dailyLimit")
        BigDecimal dailyLimit,

        @NotNull
        @DecimalMin("0.00")
        @Digits(integer = 15, fraction = 2)
        @JsonProperty("monthlyLimit")
        BigDecimal monthlyLimit,

        @NotNull
        @Min(300) @Max(1000)
        @JsonProperty("creditScore")
        Integer creditScore,

        @NotNull
        @Min(0)
        @JsonProperty("txLast24h")
        Integer txLast24h,

        @NotNull
        @Min(0)
        @JsonProperty("txLast7d")
        Integer txLast7d,

        @NotNull
        @DecimalMin("0.00")
        @Digits(integer = 12, fraction = 2)
        @JsonProperty("avgTxAmount30d")
        BigDecimal avgTxAmount30d,

        @NotNull
        @Min(0)
        @JsonProperty("secondsSinceLastTx")
        Integer secondsSinceLastTx,

        @NotNull
        @JsonProperty("isNewDevice")
        Boolean isNewDevice,

        @NotNull
        @JsonProperty("isNewBeneficiary")
        Boolean isNewBeneficiary,

        @NotNull
        @Min(0)
        @JsonProperty("failedAttempts24h")
        Integer failedAttempts24h,

        @NotNull
        @DecimalMin("0.0") @DecimalMax("1.0")
        @JsonProperty("deviceTrustScore")
        Double deviceTrustScore
) {

}
