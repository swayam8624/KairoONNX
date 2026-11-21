# KairoONNX — guide to the importer, optimizer, and runtime

**Purpose:** turn selected ONNX model data into validated, executable Kairo tensor computations. ONNX defines a much wider ecosystem; the supported operator set is deliberately bounded and must be established from the actual implementation, not the `OpKind` enum alone.

## Read in this order

1. [README](../README.md), [CMake](../CMakeLists.txt) — supported model subset, compiler configuration, tests.
2. [src/KairoONNX.cppm](../src/KairoONNX.cppm) — `TensorInfo`, `Node`, `Graph`, operator metadata; protobuf-wire reading, typed initializer handling, `ImportModelBytes`, `ValidateGraph`.
3. [src/KairoONNXRuntime.cppm](../src/KairoONNXRuntime.cppm) — `RuntimeValue` variants, input bindings, topological evaluation, native `KairoMath::Tensor` operators and output extraction.
4. [src/KairoONNXOptimize.cppm](../src/KairoONNXOptimize.cppm) — metadata propagation, constant folding using the evaluator, and pruning unreachable graph values.
5. [tests/smoke.cpp](../tests/smoke.cpp) — golden reference fixture comparisons and malformed input handling. Observe that `NDEBUG` is undefined so C assertions still execute in Release acceptance tests.
6. [tools/generate_reference_fixture.py](../tools/generate_reference_fixture.py) — provenance and procedure for the model/input/output fixtures.
7. [tests/fixtures](../tests/fixtures/) — reference MLP, CNN and transformer model bytes and expected Float32 outputs. Binary fixtures are test data, not source comments or blanket MIT-licensed code.

## Cross-file call trace

`ImportModelBytes(sourceBytes)` in the parser → bounded protobuf decode / graph metadata → `ValidateGraph` checks → runtime value resolution and operator evaluation in `Kairo.ONNX.Runtime` → `ExecutionResult.outputs`. An optional optimization pass in `Kairo.ONNX.Optimize` may use runtime evaluation to fold constant subgraphs before graph execution. `KairoMath::Tensor` supplies actual numerical storage and operations; ONNX graph names must not be mistaken for C++ variable lifetimes.

## Key invariants

An ONNX node may refer only to available upstream graph inputs, initializer values or prior node outputs. Tensor shapes, dtypes and attributes must satisfy the supported native implementation. A declared operation is not evidence that every opset variation is supported. Parsing, optimization and execution can each fail for different reasons: malformed input bytes, incompatible graph semantics, or unsupported execution.

## Exercises

1. Trace one `Add` from wire-format node to two tensor inputs and final output buffer.
2. Explain how constant folding evaluates a node without changing results at the graph boundary.
3. Compare fixture outputs element-wise with appropriate numerical tolerances.
4. Corrupt a protobuf length/initializer payload and identify its error boundary.
5. Explain why names, symbolic dimensions and runtime tensor shapes need separate validation.

## Permissions

See [LICENSE](../LICENSE) for original repository code and documentation. Fixtures and files generated from external frameworks/model assets require an independent provenance review; their copyright/permissions are **not** implied by the source license. See [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).
