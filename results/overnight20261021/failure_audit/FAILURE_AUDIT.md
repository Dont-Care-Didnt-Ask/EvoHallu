# Overnight candidate failure audit

**all_planned_jobs_complete**: 1500 persisted candidate rows, 841 pipeline-valid; 1500/1500 requests have saved generation outcomes.

Counts describe the current backup snapshot. Unissued requests and responses awaiting a persisted evaluation result are not counted as failures. Valid constant detectors are separate from invalid programs.

## Observed validity bottleneck

The largest observed failure category is **Array/ragged/axis/index/alignment handling or nonfinite numeric output**: 420/659 invalid ledger rows. This identifies the present bottleneck; it is not a causal comparison between unfinished arms.

| Category | Candidate rows |
| --- | ---: |
| Pipeline-valid and nonconstant on saved training rows | 785 |
| Array/ragged/axis/index/alignment handling or nonfinite numeric output | 420 |
| Missing import/name, wrong import location or invalid library/object API | 93 |
| Estimator inheritance/constructor interface mismatch | 66 |
| Pipeline-valid but exactly constant on all saved training rows | 56 |
| Failure not safely assigned to the preceding categories | 55 |
| Syntax/extraction failure with an explicit length/max-token finish reason | 8 |
| Syntax, indentation or class/code extraction failure without explicit truncation evidence | 6 |
| Explicit candidate evaluation timeout | 5 |
| Pipeline rejected an exact duplicate generated program | 4 |
| Generation request timeout or connection timeout | 2 |

## Phase, policy and writer

| Phase / policy | Writer | Saved outcomes / budget | Ledger rows | Valid | Invalid | Valid constant |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| evolution_screen / compositional | Qwen/Qwen3.5-9B | 180/180 | 180 | 78 | 102 | 1 |
| evolution_screen / diverse | Qwen/Qwen3.5-9B | 180/180 | 180 | 103 | 77 | 12 |
| evolution_screen / neutral | Qwen/Qwen3.5-9B | 180/180 | 180 | 98 | 82 | 7 |
| evolution_screen / original | Qwen/Qwen3.5-9B | 180/180 | 180 | 101 | 79 | 1 |
| feedback_control / neutral | Qwen/Qwen3.5-9B | 180/180 | 180 | 88 | 92 | 9 |
| interface_control / neutral | Qwen/Qwen3.5-9B | 60/60 | 60 | 29 | 31 | 1 |
| prompt_screen / compositional | Qwen/Qwen3.5-9B | 30/30 | 30 | 11 | 19 | 0 |
| prompt_screen / diverse | Qwen/Qwen3.5-9B | 30/30 | 30 | 23 | 7 | 0 |
| prompt_screen / neutral | Qwen/Qwen3.5-9B | 30/30 | 30 | 19 | 11 | 2 |
| prompt_screen / original | Qwen/Qwen3.5-9B | 30/30 | 30 | 19 | 11 | 1 |
| sampler_control / neutral | Qwen/Qwen3.5-9B | 180/180 | 180 | 100 | 80 | 4 |
| writer_confirmation / diverse | Qwen/Qwen3.8-27B | 60/60 | 60 | 56 | 4 | 1 |
| writer_confirmation / original | Qwen/Qwen3.8-27B | 60/60 | 60 | 52 | 8 | 0 |
| writer_generalization / diverse | zai-org/GLM-4.7-Flash | 60/60 | 60 | 36 | 24 | 11 |
| writer_generalization / original | zai-org/GLM-4.7-Flash | 60/60 | 60 | 28 | 32 | 6 |

## Actual error examples

All original error strings, source ledger/line hashes and generated-code hashes are retained in JSON. Most common exact messages:

Count 96:

```text
ValueError: Input contains NaN.
```

Count 82:

```text
ValueError: setting an array element with a sequence. The requested array has an inhomogeneous shape after 1 dimensions. The detected shape was (512,) + inhomogeneous part.
```

Count 56:

```text
TypeError: object.__init__() takes exactly one argument (the instance to initialize)
```

Count 51:

```text
AttributeError: 'list' object has no attribute 'shape'
```

Count 28:

```text
TypeError: list indices must be integers or slices, not tuple
```

Count 23:

```text
ValueError: setting an array element with a sequence.
```

Count 18:

```text
ValueError: operands could not be broadcast together with shapes (27,) (28,) 
```

Count 18:

```text
ImportError: cannot import name 'Estimator' from 'sklearn.base' (/workspace/mod1/.venv/lib/python3.12/site-packages/sklearn/base.py)
```

Count 15:

```text
ImportError: cannot import name 'entropy' from 'scipy.special' (/workspace/mod1/.venv/lib/python3.12/site-packages/scipy/special/__init__.py)
```

Count 15:

```text
ValueError: Input contains infinity or a value too large for dtype('float32').
```

## Truncation and constant-output interpretation

Observed syntax/extraction failures without an explicit length finish reason: **6**. They are not attributed to the token cap.

A constant training prediction vector is a valid but degenerate detector, not an import/parser/runtime failure. A 50-AUROC result does not establish constant predictions; the actual saved vector is checked. Missing score artifacts remain unknown.

No candidate source was repaired, no model was called, and no held-out dataset score was inspected. This audit measures implementation validity and numeric degeneracy, not detection quality.
