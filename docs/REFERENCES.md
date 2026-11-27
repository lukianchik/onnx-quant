# References / литература

Related work in post-training quantization and ONNX graph representation. The v4 experiments compare candidate-selection policies against the project's MinMax baseline; direct comparisons with the published methods are outside the evaluated scope.

## Post-training quantization

1. M. Nagel, R. A. Amjad, M. van Baalen, C. Louizos, T. Blankevoort. **Up or Down? Adaptive Rounding for Post-Training Quantization.** ICML, PMLR 119, 7197–7206, 2020. [Paper](https://proceedings.mlr.press/v119/nagel20a.html). Adaptive weight rounding informed by data and reconstruction objectives; relevant to activation-aware candidate construction.
2. E. Frantar, S. Ashkboos, T. Hoefler, D. Alistarh. **GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers.** ICLR, 2023. [Paper](https://arxiv.org/abs/2210.17323). Approximate second-order information for weight quantization; relevant as a comparison point for activation-dependent errors.
3. A. Zhang, Z. Yang, N. Wang, Y. Qi, J. Xin, X. Li, P. Yin. **COMQ: A Backpropagation-Free Algorithm for Post-Training Quantization.** 2024 preprint. [Paper](https://arxiv.org/abs/2403.07134). Coordinate-wise minimization of layer reconstruction error over integer codes and scaling factors; closely related to the local integer-descent objective.

## ONNX representation and execution

4. ONNX. **DequantizeLinear operator specification.** [Specification](https://onnx.ai/onnx/operators/onnx__DequantizeLinear.html). Defines the affine dequantization operation and per-axis scaling semantics used by the exported weight-only graphs.
5. ONNX Runtime. **Quantize ONNX models.** [Documentation](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html). Context for quantized graph representations, supported execution paths and their performance limitations.

## Project-specific evidence

- [Mathematical objectives and guards](MATHEMATICS.md).
- [Research code and protocol](../research/README.md).
- [Final v4 benchmark](../research/results/12_benchmark_v4_final/README.md).

A controlled comparison with AdaRound, GPTQ and COMQ would require consistent bit widths, supported layers, calibration and held-out datasets, and inference environments. No such head-to-head comparison is included in the current release.
