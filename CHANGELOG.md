# Changelog

## v1.0.0-rc3

- Support constant FP16 weights in all four modes, using FP32 scales and restoring FP16 at export.
- Report source weight types and combined FP32/FP16 parameter coverage.
- Add FP16 graph, calibration and inference regression tests.

## v1.0.0-rc2 — Research Preview

- Public documentation, licensing and security guidance.
- GitHub Actions regression tests and repository presentation.
- Annotated references and release notes.
- No changes to quantization algorithms.

See [release notes](docs/releases/v1.0.0-rc2.md).

## v1.0.0-rc1 — Research Preview

- Four weight-only INT8 modes: Standard, Fast, Balanced and Quality.
- Per-channel quantization of supported MatMul, Gemm and Conv weights.
- Robust candidate selection, progressive evaluation and bounded global search.
- Local Gradio interface, ONNX validation, dataset evaluation and JSON reports.
- Docker-based deployment and separate development configuration.
- Historical v4 benchmark and Digits CNN smoke-test evidence.

This release is a research preview; see the [README](README.en.md#limitations) for limitations.
