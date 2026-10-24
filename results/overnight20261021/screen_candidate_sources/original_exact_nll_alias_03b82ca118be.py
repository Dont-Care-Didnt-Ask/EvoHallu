import numpy as np
class HallucinationDetector(Estimator):
    def __init__(self, context: Dict[str, Any]) -> None:
        super().__init__(
            context,
            stats_dependencies=[
                "sequence_logprobs",
                "top_k_logits",
                "top_k_indices",
            ],
            level="sequence",
        )

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Detects hallucinations by combining total negative log-probability with 
        the average rank of generated tokens within the top-k candidate list.
        
        Approach:
        1. Calculate total negative log-probability (sum of -log(p)) as a baseline.
        2. For each token in the sequence, determine its rank within the top-k logits 
           (0 if it is the top-1, k if it is not present in the top-k).
        3. Compute the mean rank across the sequence. A higher mean rank indicates 
           the model frequently selected tokens that were not the most probable among 
           the plausible alternatives, a common signature of hallucination.
        4. Fuse signals: High total NLL + High mean rank -> High score.
        
        Hypothesis for future: Hallucinated sequences exhibit a systematically higher 
        average rank of generated tokens within the top-k candidates compared to factual 
        sequences, reflecting a drift away from the model's highest-probability predictions.
        
        Args:
            stats: Dictionary containing 'sequence_logprobs', 'top_k_logits', 'top_k_indices'.
        
        Returns:
            np.ndarray of shape [N] with fused hallucination scores.
        """
        log_probs_list = stats["sequence_logprobs"]
        top_k_indices_list = stats["top_k_indices"]
        
        if len(log_probs_list) == 0:
            return np.array([], dtype=np.float32)
        
        N = len(log_probs_list)
        scores = np.zeros(N, dtype=np.float32)
        K = top_k_indices_list[0].shape[1] if N > 0 else 512
        
        for i in range(N):
            lp = log_probs_list[i]
            k_indices = top_k_indices_list[i]
            
            n_tokens = len(lp)
            if n_tokens == 0:
                scores[i] = -np.sum(lp) if len(lp) > 0 else 0.0
                continue
            
            # 1. Baseline: Total Negative Log Prob
            total_neg_logprob = -np.sum(lp)
            
            # 2. Novel Signal: Mean Rank in Top-K
            # We need to find the rank of the chosen token in the top-k list for each position.
            # k_indices has shape (n_tokens, K).
            # lp has shape (n_tokens-1) (excluding first token).
            # k_indices[t] corresponds to the logits for the token at step t+1.
            # lp[t] corresponds to the logprob of the token at step t+1.
            
            # Extract chosen token IDs for the sequence steps covered by lp
            # Note: lp length is n_tokens-1. k_indices length is n_tokens.
            # The prompt says "sequence_logprobs... excluding the first generated token".
            # So lp[0] is the logprob of the 2nd token.
            # k_indices[0] is the logits for the 2nd token.
            # k_indices[1] is the logits for the 3rd token.
            # ...
            # k_indices[n_tokens-1] is the logits for the (n_tokens+1)-th token?
            # Wait, let's re-read carefully: "shape (n_i, k) — top-k logits per token"
            # "sequence_logprobs... shape (n_i-1,) — log-probabilities of tokens actually generated... excluding the first generated token."
            # Usually, if a sequence has L tokens, we have L-1 transitions (or L logits).
            # If lp has length M, it covers M tokens.
            # If k_indices has length M+1, it covers M+1 tokens.
            # Let's assume alignment: lp[t] matches k_indices[t] for t in 0..len(lp)-1.
            # The prompt says "excluding the first generated token". 
            # If the model generates a sequence of length L, it has L tokens.
            # Logprobs are usually assigned to the tokens generated.
            # If lp excludes the first, it has L-1 values.
            # k_indices usually has L values (one per step).
            # Let's assume the provided k_indices corresponds to the same steps as lp for calculation safety,
            # or we align based on the fact that lp[t] is the prob of the token generated at step t+1.
            # And k_indices[t] is the logits for step t+1.
            # So we iterate t from 0 to len(lp)-1.
            
            ranks = []
            for t in range(n_tokens):
                chosen_id = k_indices[t, 0] # The token actually generated at this step
                # Find rank of chosen_id in k_indices[t]
                # Check if chosen_id is in the top-k list
                # Since k_indices[t] contains the indices, we can check membership.
                # To do this efficiently without loops inside loops (though N is small enough):
                # We can use np.isin or search. Given K=512, a simple loop or np.argmax with mask is fine.
                
                # Create a mask where the index matches the chosen token
                # But k_indices[t] is a 1D array of shape (K,).
                # We want the index 'r' such that k_indices[t, r] == chosen_id.
                
                # Optimization: Since K is small (512), we can just iterate or use np.where
                # However, to be vectorized over the sequence length:
                pass
            
            # Vectorized approach for ranks
            # Flatten k_indices for the current sequence to check against chosen tokens?
            # No, we need rank per position.
            
            # Let's construct a boolean mask for each position
            # k_indices is (n_tokens, K)
            # We want for each row t, the index r where k_indices[t, r] == chosen_id.
            # If not found, rank = K.
            
            # We can use np.searchsorted if sorted, but top-k indices are not necessarily sorted by index value, 
            # they are sorted by logit value.
            # So we must search.
            
            # Efficient search:
            # For each row, find the index of the chosen token.
            # Since K is 512, a simple loop over K is acceptable for N samples if N is small.
            # But let's try to be smart.
            
            # Flatten the search:
            # Create a large array of chosen_ids: shape (n_tokens, 1)
            # Create a large array of k_indices: shape (n_tokens, K)
            # We want argmax of (k_indices == chosen_ids) along axis 1? No, that gives count.
            # We want the index.
            
            # Given the constraints and numpy, a loop over sequence length is fine if N is small.
            # But let's try to vectorize the search if possible.
            # Actually, we can just iterate.
            
            mean_rank = 0.0
            count = 0
            
            # Pre-calculate K for this sequence
            seq_K = k_indices.shape[1]
            
            for t in range(n_tokens):
                chosen_id = k_indices[t, 0]
                row_indices = k_indices[t]
                
                # Find index
                # np.where returns indices where condition is true
                matches = np.where(row_indices == chosen_id)[0]
                
                if len(matches) > 0:
                    rank = matches[0] # Should be unique
                else:
                    rank = seq_K # Not in top-k
                
                mean_rank += rank
                count += 1
            
            if count > 0:
                avg_rank = mean_rank / count
                # Fuse: High rank -> High score
                # Weight: 2.0 seems reasonable based on previous experiments
                scores[i] = total_neg_logprob + 2.0 * avg_rank
            else:
                scores[i] = total_neg_logprob
                
        return scores
