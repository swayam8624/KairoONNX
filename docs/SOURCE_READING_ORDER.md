# Teaching source map — KairoONNX
**Deep source walkthrough:** [ONNX import, graph validation and CPU execution](ONNX_MODEL_TO_TENSOR_EXECUTION.md).

Bounded ONNX graph importer and runtime interoperability, not universal ONNX execution.

Read these source links in order, identify the caller and callee for each public API, then check the error/ownership contract and its tests.

## Detailed call-by-call reading path (reconciled from teaching branch)

1. [`src/KairoONNX.cppm`](../src/KairoONNX.cppm) — `TensorInfo`, `Node`, `Graph`, bounded wire decoding, `ImportModelBytes`, `ValidateGraph` and initializer parsing. Learn which metadata is trusted only *after* validation.
2. [`src/KairoONNXRuntime.cppm`](../src/KairoONNXRuntime.cppm) — named `RuntimeValue` bindings, execution in node order, supported `KairoMath::Tensor` operations and output reconstruction.
3. [`src/KairoONNXOptimize.cppm`](../src/KairoONNXOptimize.cppm) — metadata inference, constant folding (which calls the evaluator), and pruning graph nodes that cannot affect public outputs.
4. [`tests/smoke.cpp`](../tests/smoke.cpp) — golden-output fixtures, unsupported/bad graph cases and active assertions even in `Release` builds with `NDEBUG` normally set.
5. [`tools/generate_reference_fixture.py`](../tools/generate_reference_fixture.py) — how reference models/inputs/outputs are generated; inspect framework versions and reproducibility.
6. [`tests/fixtures/`](../tests/fixtures/) — binary fixtures for selected MLP, CNN and other bounded cases. Fixtures have separate provenance from the original source license.

**Cross-file trace:** model bytes → bounded protobuf decoder → validated graph and initializer records → optional optimizer → runtime input/initializer binding → per-node tensor execution → named outputs. A declared ONNX op or symbolic shape is *not* proof that every opset variant or actual dynamic dimension is supported. The importer, optimizer and interpreter may fail for distinct reasons.

**Student exercises:** follow one `Add` from protobuf bytes to output; compare fold-before/fold-after outputs; corrupt a wire length and track the diagnostic; demonstrate why duplicate producers/topological ordering must be checked; check Float32 error tolerance on generated fixture outputs.

**Third-party note:** [the repository license](../LICENSE) governs original works; fixture/model provenance and externally sourced assets need separate review. See [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

## Reading order

1. [`README.md`](../README.md)
2. [`CMakeLists.txt`](../CMakeLists.txt)
3. [`src`](../src)
4. [`tools`](../tools)
5. [`tests`](../tests)
6. [`research`](../research)

## Repository integration and API contracts

Models/initializers are parsed into checked tensor metadata; KairoGPU/MLLibrary provide selected execution primitives. Unsupported ops and dtypes must fail explicitly.

## Teaching standards

- Comments explain invariants, decision reasons, lifetime, prerequisites and cross-file effects; they do not narrate obvious statements.
- Follow upstream interfaces before implementation details; distinguish implemented, host-verified and planned features.
- Run available tests at a pinned commit and consult [STATUS.md](../STATUS.md) / [STATUS.yaml](../STATUS.yaml) before declaring completion.
- Reuse original contributions under [LICENSE](../LICENSE), subject to third-party code and asset terms.


## Complete source-file inventory

After the curated teaching path, follow this deterministic inventory of first-party source, scripts, shaders, and tests. Grouping is a suggested learning order, not an indication that every function is already annotated. Vendored dependencies, ROMs, binaries, data, media, and generated evidence are excluded.

### Entry points and build (1)

1. [`CMakeLists.txt`](../CMakeLists.txt)

### Interfaces and modules (3)

1. [`src/KairoONNX.cppm`](../src/KairoONNX.cppm)
2. [`src/KairoONNXOptimize.cppm`](../src/KairoONNXOptimize.cppm)
3. [`src/KairoONNXRuntime.cppm`](../src/KairoONNXRuntime.cppm)

### Automation and tooling (1)

1. [`tools/generate_reference_fixture.py`](../tools/generate_reference_fixture.py)

### Verification and tests (1)

1. [`tests/smoke.cpp`](../tests/smoke.cpp)

Snapshot totals: 6 first-party source/build/test files, 23 tracked files overall. Re-audit when files are renamed or added.
