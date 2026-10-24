
class HallucinationDetector(Estimator):
    def __init__(self, context: Dict[str, Any]) -> None:
        super().__init__(
            context,
            stats_dependencies=[
                "sequence_logprobs",
            ],
            level="sequence",
        )

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        # "sequence_logprobs": List[np.ndarray] of length N, inner 1d array has length n_i - 1.
        # MUST return shape [N] where N = number of samples
        total_neg_logprob = np.array([-np.sum(sequence_logprobs) for sequence_logprobs in stats["sequence_logprobs"]], dtype=np.float32)
        return total_neg_logprob

