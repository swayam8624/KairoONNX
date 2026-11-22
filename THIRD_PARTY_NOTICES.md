# Third-party and fixture provenance

This repository contains binary `.onnx` test models and Float32 input/output reference data under `tests/fixtures/`, plus a reference-fixture generator in `tools/`.

The repository MIT license is intended for code and documentation owned by the repository's author. **It does not automatically establish a redistribution license for externally sourced models, datasets, or frameworks.** The fixture generator and its imported libraries should be inspected to record the origin, reproducibility process, and applicable licensing of each binary fixture.

Do not assert that all possible ONNX operators or backend runtimes are present merely because test fixtures include MLP/CNN/transformer examples. Keep fixture binaries intact during source documentation passes.
