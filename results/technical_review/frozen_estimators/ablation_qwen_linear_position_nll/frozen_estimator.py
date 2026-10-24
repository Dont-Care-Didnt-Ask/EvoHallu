import numpy as np
class HallucinationDetector(Estimator):
    def __init__(self, context):
        super().__init__(context, stats_dependencies=["sequence_logprobs"], level="sequence")
    def __call__(self, stats):
        result = []
        for lp in stats["sequence_logprobs"]:
            nll = -np.asarray(lp, dtype=np.float64)
            weights = np.arange(1, len(nll)+1, dtype=np.float64)
            result.append(len(nll)*np.average(nll, weights=weights) if len(nll) else 0.)
        return np.asarray(result)
