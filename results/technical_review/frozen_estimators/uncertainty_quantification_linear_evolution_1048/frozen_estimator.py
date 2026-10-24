from packing.evaluate.uncertainty_quantification.supervised_baselines import compact_features
class HallucinationDetector(Estimator):
    def __init__(self, context):
        super().__init__(context, stats_dependencies=["sequence_logprobs", "token_entropies", "top_k_logits"], level="sequence")
    def __call__(self, stats):
        weights = np.array([0.0, 0.0, 0.0, 0.0, -1.2262056516765416, 0.0, 0.0, 1.4101867231035483, 0.0, 0.45776576794173196, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        scales = np.array([30.0, 100.0, 5.0, 3.0, 15.0, 10.0, 2.0, 5.0, 5.0, 5.0, 3.0, 3.0, 1.0, 5.0, 6.0, 5.0, 5.0])
        return (compact_features(stats) / scales) @ weights
