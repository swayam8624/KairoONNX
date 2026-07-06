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
- `ImportModelBytes`: parser entry point with explicit diagnostics.

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

1. Add protobuf reader or generated ONNX schema support.
2. Parse graph/tensor metadata.
3. Parse initializers into Kairo tensors.
4. Implement shape inference for the first operator set.
5. Lower graph into Kairo runtime IR.
6. Validate imported outputs against fixtures.
