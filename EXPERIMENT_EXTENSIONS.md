# Experiment extension ideas

Extensions to [the original paper](https://arxiv.org/abs/2604.03473), prioritizing small changes to the existing EvoHallu pipeline and potentially informative results. The first four tasks reflect the main proposed directions; the remainder are complementary experiments.

**Common protocol:** keep the Llama-3.1-8B feature model, cached inputs, data splits and baseline set fixed unless explicitly testing them. Select final estimators on validation data and freeze them before held-out evaluation. Use multiple evolution seeds where affordable. Report both improvement over baseline in ROC-AUC percentage points and search cost (generated/evaluated candidates, generation tokens, GPU-hours or API cost); plot best validation AUC against cumulative cost rather than relying only on a gain/cost ratio.

## 1. Benchmark newer open-weight code-generating models

Test whether newer open-weight models can design better uncertainty estimators than GPT-OSS-120B at similar or smaller scale. Shortlist 2–3 currently available coding/reasoning models after checking release dates, licensing and local serving compatibility; include a substantially smaller model to test cost efficiency. For mixture-of-experts models, report both total and active parameters.

Start with short, equal-round-budget runs using the existing prompts, then extend the most promising models. Change the **estimator-writing model**, not the model producing uncertainty inputs. Most changes should be confined to model configuration and serving/model-name handling.

**Potential result:** a newer or smaller open model outperforms GPT-OSS-120B, or achieves comparable gains at much lower cost.

## 2. Improve candidate diversity and search efficiency

Test a two-stage generator: one LLM call proposes `num_outputs_per_prompt` explicitly different method sketches, followed by one implementation call per sketch. Require different signal families or aggregation principles—not merely different coefficients.

Compare this against direct generation and a cheaper intervention: randomized, structured prompt variants. Note: the current `system_prompt_fabric()` already randomizes a baseline with an auxiliary idea, but this approach did not provide significant improvements. Apparently, this was insufficient for improving diversity.

**Potential result:** deliberate diversity improves AUC per unit compute, or reveals that additional syntactic diversity does not translate into useful prediction diversity.

## 3. Explain the Claude model ranking through candidate analysis

Analyze the reported **Sonnet 4.5 > Opus 4.5 > Opus 4.6** ranking once candidates from those runs become accessible. This task is currently **blocked on candidate access**; winning formulas alone would support only a limited analysis, not conclusions about the full search distribution.

 Manually inspect a small matched sample of successful and failed candidates, and track how each population changes over evolution. Check budgets, prompts and seeds before attributing differences to model capability. Compare method-family diversity, prediction similarity, code complexity, invalid-program rates, runtime and train–validation gaps.

**Potential result:** stronger general-purpose models may converge on similar or unnecessarily elaborate estimators, while a simpler, more diverse search generalizes better. Treat this as a hypothesis, not an assumed explanation.

## 4. Evolve supervised uncertainty estimators

Extend evolution to methods that fit labelled reference data. The code already has `task.supervised` support and training-statistics plumbing, but the active prompt describes unsupervised methods; audit the estimator interface and update the prompt and evaluation path before treating the flag as a complete supervised setup. Code changes might be required for supporting supervised methods correctly. For supervised methods, evolution run still must only store the candidate code. However, additional effort is required to ensure reproducibility between evolution run and evaluation. For evaluation, supervised methods still must only use PopQA train subset, and be trained once (evaluation code will also require some changes).

Give each candidate a fixed small tuning budget, such as five randomly sampled hyperparameter configurations. Choose hyperparameters by in-sample training AUC - it should be okay for small tuning budget. Keep the outer validation split for final candidate selection and test labels unavailable to fitting or tuning.

Compare against manually designed supervised baselines—e.g. logistic regression or Quadratic Discriminant Analysis — with comparable tuning budgets, as well as the unsupervised methods.

**Potential result:** evolution discovers a useful learned estimator, or establishes how much apparent progress disappears against properly tuned supervised baselines.

## 5. Test whether evolution is necessary

Compare the existing evolutionary search with independent sampling from the same task prompt and a baseline-seeded, non-evolutionary generator. Match candidate or compute budgets and apply the same validation-selection procedure. Reuse cached features and evaluation code; only the prompt/example-selection loop needs to change.

**Potential result:** quantify the value of evolutionary feedback over simply asking a model for many methods, and identify when evolution starts paying off.

## 6. Test transfer across feature models

Freeze a small set of selected formulas and evaluate them using uncertainty inputs from one additional, affordable feature model. Recompute inputs with consistent preprocessing and retain comparable manually designed baselines. Do not retune the formulas on the target model's test results.

**Potential result:** distinguish broadly useful uncertainty principles from formulas specialized to Llama-3.1-8B. This complements the existing cross-dataset evaluation without requiring another full evolution campaign.

## 7. Test whether complexity actually helps

For several strong evolved formulas, analyze feature importance by dropping them or using Shapley values (either only shuffling feature values and keeping a single pre-fitted model, or re-fitting the model if it is cheap). Measure AUC vs kept feature count. Formulas for analysis should likely be taken from the original paper, Table 3 (Evo-Anth-1, Evo-Anth-15, or Evo-Gpt-1, Evo-Gpt-3).

**Potential result:** obtain a compact, interpretable estimator with similar performance, or isolate a genuinely useful new ingredient. This also provides concrete evidence for the Claude complexity/diversity analysis.