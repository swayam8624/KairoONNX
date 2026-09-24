# KairoONNX v1 Status

**Frozen v1 source completion: 95%.**

- Wave: `E`
- Frozen scope: `bounded-onnx-runtime-v1`
- Source gate: `complete`
- Exact-head execution: `pending_external_runner`
- Verification gate: `cmake-build-ctest-reference-fixtures`
- Research track: `R7-support`
- Warning policy: `zero-kairo-owned-warnings`

The 95% score measures the bounded v1 implementation, integration contract,
tests/diagnostics surface and documentation. Native/platform execution evidence
is tracked separately and is never inferred from this score.

## Explicitly post-v1

- full ONNX compatibility
- all dtypes/opsets
- negative-step slicing

See `STATUS.yaml` for the machine-readable contract.
