import numpy as np
from scipy.stats import pearsonr
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
        """
        Detects hallucinations by contrasting the log-probability slope 
        between the first and last quarters of the sequence, combined with 
        the total negative log-prob sum.
        
        Approach:
        1. Compute the baseline difficulty score: Sum of Negative Log-Probabilities (NLP Sum).
        2. Split the sequence into four equal segments: First, Second, Third, Last.
        3. Calculate the Pearson correlation coefficient (slope proxy) for the First 
           segment (r_start) and the Last segment (r_end).
        4. Compute the absolute difference in slopes: delta_slope = |r_start - r_end|.
        5. Combine signals: Score = NLP_Sum + gamma * delta_slope.
        
        Hypothesis for Future: Hallucinated sequences exhibit a significant divergence 
        in the rate of log-probability decay between the beginning and end of the sequence 
        (high delta_slope), whereas factual sequences maintain a consistent decay pattern 
        (low delta_slope), reflecting a stable generation trajectory.
        
        Output shape: [N] where N is the number of samples.
        """
        N = len(stats["sequence_logprobs"])
        scores = np.zeros(N, dtype=np.float32)
        
        # Hyperparameter for weighting the structural contrast signal
        gamma = 2.0
        
        for i, seq_logprobs in enumerate(stats["sequence_logprobs"]):
            if len(seq_logprobs) < 4:
                # If sequence is too short to split into quarters, rely on baseline only
                nlp = -np.array(seq_logprobs, dtype=np.float32)
                scores[i] = np.sum(nlp)
                continue

            # 1. Baseline: Sum of Negative Log-Probabilities
            nlp_sum = np.sum(-np.array(seq_logprobs, dtype=np.float32))
            
            # 2. Split sequence into quarters
            n_tokens = len(seq_logprobs)
            quarter_size = n_tokens // 4
            
            # Define indices for First and Last quarters
            # First Quarter: [0, quarter_size)
            start_idx = 0
            end_idx = quarter_size
            
            # Last Quarter: [3*quarter_size, n_tokens)
            last_start = 3 * quarter_size
            last_end = n_tokens
            
            first_quarter = seq_logprobs[start_idx:end_idx]
            last_quarter = seq_logprobs[last_start:last_end]
            
            # 3. Compute Slope (Pearson Correlation) for First and Last Quarters
            # We use positions relative to the start of each quarter
            positions_start = np.arange(len(first_quarter))
            positions_last = np.arange(len(last_quarter))
            
            try:
                r_start, _ = pearsonr(positions_start, first_quarter)
            except Exception:
                r_start = 0.0
                
            try:
                r_end, _ = pearsonr(positions_last, last_quarter)
            except Exception:
                r_end = 0.0
            
            # 4. Compute Contrast Signal
            delta_slope = abs(r_start - r_end)
            
            # 5. Combine Signals
            # Higher NLP sum = harder sequence (potential hallucination)
            # Higher delta_slope = inconsistent decay pattern (potential hallucination)
            final_score = nlp_sum + (gamma * delta_slope)
            
            scores[i] = final_score

        return scores
