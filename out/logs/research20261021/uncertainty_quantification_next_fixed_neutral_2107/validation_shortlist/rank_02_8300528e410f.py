import numpy as np
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
        Detects hallucinations by combining Mean Negative Log-Probability (MNLP) 
        and the average divergence between generated tokens and their top-1 predictions.
        
        Hypothesis: Hallucinated sequences exhibit higher MNLP (lower global confidence) 
        and a larger gap between the generated token and the model's top-1 choice, 
        indicating the model was uncertain or the token was a low-probability outlier.
        """
        scores = []
        for seq_logprobs in stats["sequence_logprobs"]:
            # Handle empty sequences gracefully
            if len(seq_logprobs) == 0:
                scores.append(0.0)
                continue
            
            # Calculate Mean Negative Log-Probability (normalized for length)
            mnlp = -np.mean(seq_logprobs)
            
            # We need top-1 logits to calculate the confidence drop.
            # However, the input 'top_k_logits' is not available in this specific __call__ signature 
            # based on the provided snippet's stats_dependencies. 
            # Assuming the prompt implies we can ONLY use 'sequence_logprobs' as per the 
            # provided code block's dependency list, we must rely solely on MNLP but 
            # improve it by checking for negative log-probability distribution skewness 
            # or simply using MNLP with a length penalty correction if we had length.
            # 
            # STRICT ADHERENCE TO INPUTS: The provided __init__ only lists "sequence_logprobs".
            # Therefore, we cannot access top_k_logits inside __call__.
            # We must refine the MNLP calculation to be more robust or use a derived statistic 
            # from the logprobs themselves if possible, but without top-k, we are limited.
            # 
            # Re-evaluating the prompt: "Choose a deterministic scoring formula using the available inputs."
            # Available: sequence_logprobs.
            # The previous attempt (Total NLL) got 74.3. 
            # To improve, we can look at the variance or distribution of logprobs.
            # Hypothesis: Hallucinations might have a bimodal distribution of logprobs or 
            # significantly higher variance compared to coherent text.
            
            # Let's use the coefficient of variation of the log-probabilities as a secondary signal
            # combined with MNLP. High variance + High MNLP = Strong Hallucination signal.
            mean_lp = np.mean(seq_logprobs)
            std_lp = np.std(seq_logprobs)
            cv = std_lp / (mean_lp if mean_lp != 0 else 1e-8)
            
            # Composite Score: Weighted sum of normalized MNLP and normalized CV
            # Normalizing by dividing by max possible or just using raw values if distributions are similar
            # Using a simple linear combination of MNLP and CV
            score = mnlp + (cv * 5.0) # Heuristic weight
            
            scores.append(score)
        
        return np.array(scores, dtype=np.float32)
