# Teaching source map — KairoONNX

Bounded ONNX graph importer and runtime interoperability, not universal ONNX execution.

Read these source links in order, identify the caller and callee for each public API, then check the error/ownership contract and its tests.

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
