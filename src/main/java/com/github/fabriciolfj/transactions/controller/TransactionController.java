package com.github.fabriciolfj.transactions.controller;

import ai.djl.inference.Predictor;
import ai.djl.translate.TranslateException;
import com.github.fabriciolfj.transactions.dtos.ApprovalResponse;
import com.github.fabriciolfj.transactions.dtos.TransactionRequest;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.http.HttpStatus;
import org.springframework.web.bind.annotation.*;

import java.util.function.Supplier;

@RequiredArgsConstructor
@RestController
@RequestMapping("/api/v1/transactions")
public class TransactionController {

    private final Supplier<Predictor<TransactionRequest, ApprovalResponse>> predictorSupplier;

    @PostMapping
    @ResponseStatus(HttpStatus.ACCEPTED)
    public ApprovalResponse requestAssessTransaciton(@RequestBody @Valid final TransactionRequest request) throws TranslateException {
        return predictorSupplier.get().predict(request);
    }
}
