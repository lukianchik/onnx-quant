# Benchmark v4 Final — 2026-11-24

[Русский анализ](README.ru.md) · [Per-seed results](decision_per_seed.csv) · [Policy gate](decision_gate.csv) · Raw reports and logs are not included in this public snapshot.

## Execution

8 models × 3 calibration seeds (42, 123, 731) × 17 methods/ablations = 408 rows: 396 successful, 12 expected skips of legacy classification-only V2; no failed jobs. 24/24 model/seed jobs completed. All 24 manifests report CUDA inference (DEFAULT convolution algorithm), PyTorch 2.11.0+cu130, ORT 1.30.0. CUDA provider selection is not proof that every ONNX node ran on GPU. 533 of 534 discovered operator occurrences validated; autoencoder seed 123 `latent_W` excluded due to FP32 adapter mismatch. No fused LSTM recurrent matrices quantized.

## Model-level medians

Each ratio = max over outputs of MSE(candidate, FP32) / MSE(MinMax, FP32) per seed, then median across seeds. Lower is better. Not accuracy improvement.

| Model             |   Single-Robust |   Progressive |   Progressive-Global |   PG shift |   PG reduction % |
|:------------------|----------------:|--------------:|---------------------:|-----------:|-----------------:|
| autoencoder       |          1.0000 |        0.4593 |               0.4493 |     0.4838 |          55.0731 |
| efficientnet_b0   |          0.1500 |        0.1408 |               0.1389 |     0.1321 |          86.1073 |
| lstm_head         |          0.3791 |        0.3791 |               0.3791 |     0.4389 |          62.0919 |
| minilm_embeddings |          1.0000 |        0.9879 |               0.9058 |     0.9092 |           9.4230 |
| mlp               |          0.2627 |        0.1483 |               0.1483 |     0.2169 |          85.1724 |
| multi_output      |          0.4427 |        0.3553 |               0.3553 |     0.4145 |          64.4681 |
| regression        |          1.0000 |        1.0000 |               0.7470 |     0.9156 |          25.2975 |
| vit_tiny          |          0.9132 |        0.8847 |               0.8847 |     0.8921 |          11.5257 |

## Policy acceptance

| method             | pass   |   models >=5% |   mean ratio |   estimated seconds |
|:-------------------|:-------|--------------:|-------------:|--------------------:|
| Progressive-Global | True   |             8 |       0.5011 |            829.0890 |
| Single-Dev         | False  |             7 |       0.5075 |            250.4571 |
| Progressive        | True   |             6 |       0.5444 |            496.3460 |
| FullBank-Robust    | True   |             6 |       0.5446 |           1357.2062 |
| SDR-Robust         | True   |             6 |       0.5570 |            752.5016 |
| Astra-Adaptive     | True   |             6 |       0.6277 |            711.2900 |
| Paired-Robust      | True   |             5 |       0.6283 |            383.0105 |
| Single-Robust      | True   |             5 |       0.6435 |            250.1180 |

Gate from notebook cell 14: complete 8×3; same serialized size; each model's median test <=1.02, median shift <=1.05, accuracy delta >=-0.01 where available, task RMSE ratio <=1.02 where available; >=2 models with median test <=0.95. Seven of eight guarded policies passed. Progressive-Global wins by mean of model medians; Progressive is a lower-cost option; Single-Robust is a fast option. The final report is a policy comparison on this test suite, not a new held-out evaluation of the chosen winner.

## Coverage and limitations

| model             |   eligible |   validated |   quantized_share |
|:------------------|-----------:|------------:|------------------:|
| autoencoder       |          6 |           5 |            0.9624 |
| efficientnet_b0   |        246 |         246 |            0.9940 |
| lstm_head         |          3 |           3 |            0.0561 |
| minilm_embeddings |        108 |         108 |            0.4705 |
| mlp               |          6 |           6 |            0.9826 |
| multi_output      |          9 |           9 |            0.9713 |
| regression        |          6 |           6 |            0.9143 |
| vit_tiny          |        150 |         150 |            0.9878 |

See [per-seed audit](per_seed_audit.csv), [coverage](coverage.csv), [guard audit](guard_audit.csv), [sizes](size_medians.csv). Progressive-Global has zero observed test/shift MSE regressions in 24 runs; one complete rollback (autoencoder seed 123), and a -0.002778 classification accuracy delta for MLP seed 731. Global guard checks output MSE, not every downstream metric. No arbitrary-distribution guarantee. Fixed test examples across seeds; seed-level bootstrap is not an independent cross-seed confidence interval. ImageNet checkpoints on CIFAR-10 measure fidelity, not CIFAR accuracy. MiniLM measures embeddings, not retrieval quality; truncation shift may be mild. Policy preparation cost is replay estimate from a prebuilt bank plus measured replay time; calibration, full bank generation and final export/inference are separate. No proven wall-clock speedup of lazy implementation yet. Weight-only INT8 + DQ does not guarantee INT8 compute or faster inference. No SOTA comparison or verified novelty. The public evidence includes summary and per-seed tables; raw logs, model weights and activation banks are not distributed.
