package com.github.fabriciolfj.transactions.configurations;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.Shape;
import ai.djl.translate.NoBatchifyTranslator;
import ai.djl.translate.TranslatorContext;
import com.github.fabriciolfj.transactions.dtos.ApprovalResponse;
import com.github.fabriciolfj.transactions.dtos.TransactionRequest;

import static com.github.fabriciolfj.transactions.configurations.MapperOutput.*;

public class TransactionTransformer implements NoBatchifyTranslator<TransactionRequest, ApprovalResponse> {

    public static final String REQUEST_KEY   = "request";
    public static final int    FEATURE_COUNT = 21;

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

        final var mapper = new MapperInput(scalerMean, scalerScale, encoders);
        final float[] features = mapper.buildFeatureVector(request);
        final NDArray array    = ctx.getNDManager().create(features, new Shape(1, FEATURE_COUNT));

        return new NDList(array);
    }

    @Override
    public ApprovalResponse processOutput(final TranslatorContext ctx, final NDList list) {
        var probability = extractProbability(list);
        var request = extractRequest(ctx);
        var reasons = collectRiskReasons(request, probability);
        var isApproved = MapperOutput.isApproved(request, probability, threshold);

        return isApproved
                ? ApprovalResponse.approved(probability, reasons)
                : ApprovalResponse.denied(probability, pickSuggestion(request), reasons);
    }
}