# Teaching source map — KairoONNX

Bounded ONNX graph importer and runtime interoperability, not universal ONNX execution.

Read these source links in order, identify the caller and callee for each public API, then check the error/ownership contract and its tests.

## Reading order

1. [`README.md`](README.md)
2. [`CMakeLists.txt`](CMakeLists.txt)
3. [`src`](src)
4. [`tools`](tools)
5. [`tests`](tests)
6. [`research`](research)

## Repository integration and API contracts

Models/initializers are parsed into checked tensor metadata; KairoGPU/MLLibrary provide selected execution primitives. Unsupported ops and dtypes must fail explicitly.

## Teaching standards

- Comments explain invariants, decision reasons, lifetime, prerequisites and cross-file effects; they do not narrate obvious statements.
- Follow upstream interfaces before implementation details; distinguish implemented, host-verified and planned features.
- Run available tests at a pinned commit and consult [STATUS.md](STATUS.md) / [STATUS.yaml](STATUS.yaml) before declaring completion.
- Reuse original contributions under [LICENSE](LICENSE), subject to third-party code and asset terms.
