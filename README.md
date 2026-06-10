# KairoONNX

KairoONNX is the ONNX import and model-IR package for Kairo ML.

Phase 1 scope:

- internal graph IR,
- tensor metadata,
- bounded operator registry,
- explicit parser placeholder.

The first real parser should target a small inference-safe operator set before
attempting broad ONNX coverage.
