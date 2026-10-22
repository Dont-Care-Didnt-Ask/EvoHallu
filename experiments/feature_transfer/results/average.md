# Average AUROC

**Основной режим: full-chat folded AUROC (%).** Average = Mean9, равновесное среднее девяти датасетов. Жирным выделен максимум в каждой строке, включая численно равные максимумы. Порядок всех 25 методов фиксирован.

| Метод | Llama 3.1 | Gemma | OLMo |
| --- | ---: | ---: | ---: |
| SequenceLogProb | **68.74** | 68.68 | 65.26 |
| SequenceLogProbBottom | **67.95** | 64.47 | 65.26 |
| MinimalLogProb | 51.49 | **57.17** | 56.82 |
| MeanTokenEntropy | 59.91 | **64.31** | 58.78 |
| Perplexity | 64.16 | 65.82 | **66.47** |
| StdTokenEntropy | 54.52 | **57.75** | 55.98 |
| MeanStdTokenEntropy | 58.71 | **64.90** | 57.28 |
| QuartileEntropyDiffNormalized | 57.13 | **68.64** | 59.00 |
| QuartileEntropyDiff | 57.13 | **68.64** | 59.00 |
| PositionWeightedGap | 66.51 | **66.70** | 66.22 |
| PositionWeightedGapNormalized | 66.51 | **66.70** | 66.22 |
| Gap | 63.32 | 65.32 | **65.94** |
| QuartileEntropyDiff_PositionWeightedGap | 68.51 | **70.06** | 65.95 |
| QuartileEntropyDiff_PositionWeightedGap_TopKMassDecay | 65.94 | **69.63** | 64.18 |
| EntropySlope | 56.64 | **68.47** | 59.13 |
| LogProbSlope | **70.62** | 68.37 | 63.00 |
| ExpWeightedNegLogProbWithTop1 | **70.78** | 69.13 | 67.06 |
| ExpWeightedNegLogProb | **70.60** | 68.93 | 66.80 |
| PositionWeightedGap_EntropySlope_TopKMassInv | 66.83 | **71.56** | 64.02 |
| LogProbSlope_OutlierScore | **69.65** | 65.25 | 60.65 |
| SequenceLogProb_OutlierScore | **68.74** | 68.69 | 65.26 |
| LogProbSlope_TopBottomRatio | **70.64** | 68.38 | 63.01 |
| SequenceLogProb_TopBottomRatio | **68.74** | 68.68 | 65.26 |
| LogProbSlope_BottomMassAvg | **70.68** | 68.38 | 63.01 |
| SequenceLogProb_BottomMassAvg | **68.74** | 68.68 | 65.26 |
