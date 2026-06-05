package com.github.fabriciolfj.transactions.dtos;

import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.time.Instant;
import java.util.List;

@JsonInclude(JsonInclude.Include.NON_NULL)
public record ApprovalResponse(
        @JsonProperty("approved")
        boolean approved,

        @JsonProperty("status")
        ApprovalStatus status,

        @JsonProperty("probability")
        double probability,

        @JsonProperty("riskLevel")
        RiskLevel riskLevel,

        @JsonProperty("suggestion")
        String suggestion,

        @JsonProperty("reasons")
        List<String> reasons,

        @JsonProperty("evaluatedAt")
        Instant evaluatedAt) {

    public static ApprovalResponse approved(double probability, List<String> reasons) {
        var riskLevel = RiskLevel.from(probability);
        var status = riskLevel == RiskLevel.LOW ? ApprovalStatus.APROVADO
                : ApprovalStatus.APROVADO_COM_RESTRICAO;
        return new ApprovalResponse(true, status, probability, riskLevel,
                null, reasons, Instant.now());
    }

    public static ApprovalResponse denied(double probability,
                                          String suggestion,
                                          List<String> reasons) {
        return new ApprovalResponse(false, ApprovalStatus.NEGADO, probability,
                RiskLevel.from(probability), suggestion,
                reasons, Instant.now());
    }
}