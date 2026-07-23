# KairoONNX

KairoONNX is the model import and graph-IR package for the Kairo ML stack. Its
job is to bring external models into Kairo deliberately, operator by operator,
without turning the core runtime into an unbounded ONNX compatibility project.

## Problem

ONNX is broad. A serious importer must handle:

- graph topology,
- tensor metadata,
- initializer weights,
- opset/version differences,
- attributes,
- shape inference,
- unsupported operator diagnostics,
- runtime lowering into native kernels.

Trying to support all of ONNX at once leads to a fragile importer that silently
misinterprets models.

## Solution

KairoONNX defines a bounded import surface:

- `TensorInfo`: name, element type, shape.
- `Node`: op, inputs, outputs, attributes.
- `Graph`: inputs, outputs, initializers, nodes.
- `OpKind`: explicit supported operator enum.
- `ValidateGraph`: diagnostics before lowering.
- `InferElementwiseOutput`: first shape-inference primitive.
- `ImportModelBytes`: bounded protobuf-wire reader for `ModelProto`, graph
  topology, node input/output/op metadata, and initializer identity, element
  type, dimensions, and `raw_data` payload metadata,
  with explicit malformed-wire and unsupported-op diagnostics.

The first real parser should target a limited inference set:

- `Gemm`, `MatMul`
- `Add`, `Sub`, `Mul`, `Div`
- `Relu`, `Softmax`
- `Conv`, `MaxPool`
- `Flatten`, `Reshape`, `Transpose`
- `LayerNormalization`

## Where It Connects

- `KairoMath::Tensor`: stores imported weights and tensors.
- `MLLibrary`: lowers ONNX graphs into native inference graphs.
- `KairoTransformers`: consumes ONNX-imported transformer weights later.
- `KairoGPU` and `KairoSIMD`: execute lowered kernels through backend dispatch.

## Build

```sh
cmake -S . -B build -G Ninja -DCMAKE_CXX_COMPILER=/opt/homebrew/opt/llvm/bin/clang++
cmake --build build
ctest --test-dir build --output-on-failure
./build/KairoONNXSmoke
```

## Roadmap

1. Convert supported initializer payloads into Kairo tensors.
2. Parse node attributes and opset/version semantics.
3. Implement shape inference for the first operator set.
4. Lower graph into Kairo runtime IR.
5. Validate imported outputs against fixtures.
