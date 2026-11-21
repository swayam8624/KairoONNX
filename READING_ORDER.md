# KairoONNX — linked source learning order

Study ONNX graph import and tensor contracts before graph optimizations and bounded runtime execution. This project intentionally does not promise arbitrary ONNX operator/dtype coverage; unsupported graphs should fail explicitly.

## APIs and neighbors

Follow model file -> parsed graph/operator/initializer -> validation -> runtime tensor execution, while tracking shapes, dtypes and numerical reference comparisons. KairoTransformers and the ML tensor runtime are downstream consumers; optimizer transformations must preserve evaluated semantics.

## Read these source files (5)

1. [`src/KairoONNX.cppm`](src/KairoONNX.cppm) — inspect how public API inputs/outputs, state and error cases interact with neighboring implementations and tests.
2. [`src/KairoONNXRuntime.cppm`](src/KairoONNXRuntime.cppm) — inspect how public API inputs/outputs, state and error cases interact with neighboring implementations and tests.
3. [`src/KairoONNXOptimize.cppm`](src/KairoONNXOptimize.cppm) — inspect how public API inputs/outputs, state and error cases interact with neighboring implementations and tests.
4. [`tests/smoke.cpp`](tests/smoke.cpp) — inspect how public API inputs/outputs, state and error cases interact with neighboring implementations and tests.
5. [`tools/generate_reference_fixture.py`](tools/generate_reference_fixture.py) — inspect how public API inputs/outputs, state and error cases interact with neighboring implementations and tests.

Read [README.md](README.md) for the toolchain and [KairoGameEngine](https://github.com/swayam8624/KairoGameEngine) for exact-head ecosystem acceptance. Original source is [MIT licensed](LICENSE) for learning, modification and continued development, without relicensing dependencies/fixtures. This guide is a documented navigation plan, not comprehensive in-file comment coverage or proof of correctness.
