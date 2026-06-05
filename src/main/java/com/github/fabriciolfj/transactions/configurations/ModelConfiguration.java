package com.github.fabriciolfj.transactions.configurations;

import ai.djl.inference.Predictor;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import com.fasterxml.jackson.databind.json.JsonMapper;
import com.github.fabriciolfj.transactions.dtos.ApprovalResponse;
import com.github.fabriciolfj.transactions.dtos.TransactionRequest;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import java.io.InputStream;
import java.util.Objects;
import java.util.function.Supplier;

@Configuration
public class ModelConfiguration {

    private static final String MODEL_FILE  = "financial_approval.onnx";
    private static final String DICT_FILE   = "feature_dict.json";

    @Bean
    public JsonMapper jsonMapper() {
        return JsonMapper.builder().build();
    }

    @Bean
    public FeatureDictionary featureDictionary() throws Exception {
        try (InputStream is = resourceStream(DICT_FILE)) {
            return jsonMapper().readValue(is, FeatureDictionary.class);
        }
    }

    @Bean
    public Criteria<TransactionRequest, ApprovalResponse> criteria(
            final FeatureDictionary featureDictionary) throws Exception {

        final String modelLocation = Objects
                .requireNonNull(Thread.currentThread().getContextClassLoader().getResource(MODEL_FILE),
                        "Model not found in classpath: " + MODEL_FILE)
                .toExternalForm();

        return Criteria.builder()
                .setTypes(TransactionRequest.class, ApprovalResponse.class)
                .optModelUrls(modelLocation)
                .optTranslator(new TransactionTransformer(featureDictionary))
                .optEngine("OnnxRuntime")
                .build();
    }

    @Bean
    public ZooModel<TransactionRequest, ApprovalResponse> model(
            final Criteria<TransactionRequest, ApprovalResponse> criteria) throws Exception {
        return criteria.loadModel();
    }

    @Bean
    public Supplier<Predictor<TransactionRequest, ApprovalResponse>> predictor(
            final ZooModel<TransactionRequest, ApprovalResponse> model) {
        return model::newPredictor;
    }

    private static InputStream resourceStream(final String name) {
        return Objects.requireNonNull(
                Thread.currentThread().getContextClassLoader().getResourceAsStream(name),
                "Resource not found in classpath: " + name);
    }
}