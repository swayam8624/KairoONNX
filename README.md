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
  topology, opset imports, typed value shapes (including symbolic dimensions),
  node input/output/op metadata, scalar and packed attributes, and initializer
  identity, element type, dimensions, and `raw_data` payload metadata,
  with explicit malformed-wire and unsupported-op diagnostics.
- `Float32InitializerTensor`: converts validated little-endian Float32
  `raw_data` initializer payloads into owned `KairoMath::Tensor` values.
- `Kairo.ONNX.Runtime`: validates topological value availability, materializes
  Float32 and Int64 initializers, and executes native Kairo tensors.

Runtime lowering currently covers:

- NumPy-broadcast `Add`, `Sub`, `Mul`, and `Div`
- rank-2/batched `MatMul` and `Gemm`
- `Relu`, `Gelu`, `Sigmoid`, final-axis `Softmax`
- NCHW/OIHW `Conv` with bias and 2D `MaxPool`
- `Flatten`, `Reshape`, and arbitrary-rank `Transpose`
- final-axis `LayerNormalization`
- `Gather`, positive-step `Slice`, and `Concat`

The conformance smoke suite executes representative MLP, CNN, pooling, and
indexing graphs and compares exact expected outputs. Missing feeds,
topologically unavailable values, duplicate producers, unsupported opsets, and
incompatible shapes fail before graph outputs are returned.

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

## Remaining Conformance Work

1. Convert typed TensorProto fields and Float16/BFloat16 initializers.
2. Add grouped/dilated convolution and negative-step slicing.
3. Add graph optimization passes: constant folding and dead-value elimination.
4. Run exported ONNX fixture outputs against a trusted external runtime.
5. Expand opset-specific semantic checks instead of using one bounded 7-21
   compatibility range.
