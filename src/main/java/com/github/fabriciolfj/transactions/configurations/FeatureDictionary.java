package com.github.fabriciolfj.transactions.configurations;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.List;
import java.util.Map;


@JsonIgnoreProperties(ignoreUnknown = true)
public record FeatureDictionary(

        @JsonProperty("version")
        String version,

        @JsonProperty("threshold")
        float threshold,

        @JsonProperty("input_name")
        String inputName,

        @JsonProperty("output_name")
        String outputName,

        @JsonProperty("scaler")
        Scaler scaler,

        @JsonProperty("label_encoders")
        LabelEncoders labelEncoders

) {
    @JsonIgnoreProperties(ignoreUnknown = true)
    public record Scaler(

            @JsonProperty("mean")
            List<Double> mean,

            @JsonProperty("scale")
            List<Double> scale,

            @JsonProperty("n_numeric_features")
            int nNumericFeatures

    ) {
        public float[] meanAsFloatArray() {
            return toFloatArray(mean);
        }

        public float[] scaleAsFloatArray() {
            return toFloatArray(scale);
        }

        private static float[] toFloatArray(final List<Double> values) {
            final float[] result = new float[values.size()];
            for (int i = 0; i < values.size(); i++) {
                result[i] = values.get(i).floatValue();
            }
            return result;
        }
    }
    @JsonIgnoreProperties(ignoreUnknown = true)
    public record LabelEncoders(

            @JsonProperty("transaction_type")
            EncoderEntry transactionType,

            @JsonProperty("merchant_category")
            EncoderEntry merchantCategory,

            @JsonProperty("country")
            EncoderEntry country,

            @JsonProperty("account_tier")
            EncoderEntry accountTier

    ) {}

    @JsonIgnoreProperties(ignoreUnknown = true)
    public record EncoderEntry(

            @JsonProperty("mapping")
            Map<String, Integer> mapping

    ) {
        public float encode(final String value) {
            return mapping.getOrDefault(value, 0).floatValue();
        }
    }
}