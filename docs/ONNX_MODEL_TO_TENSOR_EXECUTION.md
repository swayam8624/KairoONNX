# KairoONNX — from untrusted model bytes to bounded tensor execution

This guide walks the actual three-module path:
[`KairoONNX.cppm`](../src/KairoONNX.cppm) (import/graph),
[`KairoONNXOptimize.cppm`](../src/KairoONNXOptimize.cppm)
(static optimization), and
[`KairoONNXRuntime.cppm`](../src/KairoONNXRuntime.cppm)
(CPU interpreter). A model file being syntactically parseable does **not**
mean the runtime supports its operators, opset, shapes or dtypes.

## 1. Parse a protocol buffer safely

`ImportModelBytes` reads a subset of ONNX protobuf wire types into
`Graph` containing named `TensorInfo` initializers, inputs/outputs,
nodes, attributes and opset information. The bounded reader must
validate varints, nested field lengths, unknown fields and malformed
wire types before building trusted graph state.

`Float32InitializerTensor` checks that the raw byte length equals
the expected element count multiplied by Float32 width and rejects
overflowing shapes. It explicitly requires little-endian data
interpretation. Loading malformed bytes must never be treated as
a request to allocate an unbounded tensor.

## 2. Validate the graph as a directed computation

`ValidateGraph` checks the model's graph contracts, including
uniqueness of produced value names and supported operators. A node
depends on named value producers; an unavailable input is an
error, not a zero/identity tensor to be invented by the runtime.
`OpKind` names a bounded operation surface, not the entire
standard's operator catalogue.

**Exercise:** import a known small fixture; then duplicate a node output,
remove a producer, mismatch initializer byte count, and introduce
an unsupported op. Record import/validation results separately
from numerical execution.

## 3. Inference shapes and constant folding

`InferStaticShapes` reasons about axes and broadcasting, MatMul,
Gemm, reshaping, Gather, Slice, Concat, Transpose, NCHW Conv and
MaxPool constraints. Known dimensions must remain consistent;
unknowns cannot be silently replaced by invented constants.

`ConstantFold` evaluates only eligible static subgraphs using the
supported CPU interpreter and publishes the output as a valid
initializer. `EliminateDeadValues` prunes nodes unreachable from
the named output values. Optimizing an imported model should leave
observable inference results equivalent to the unoptimized graph.

## 4. Runtime bindings, layout and dtype

`ExecuteGraph` uses named CPU `Tensor<float>` and
`Tensor<int64_t>` bindings. It combines validated inputs with
initializers, dispatches each supported operation, and returns
a map of named outputs/intermediate values.

Layout is critical: ONNX Conv operates on **NCHW** images with
**OIHW** weights. Some KairoMath tensor helpers use **NHWC**
images and **OHWI** filters. Reusing one kernel without explicit
axis conversion produces wrong output even if every dimension
is numerically compatible.

The current ONNX Conv implementation has restricted 2D
group/dilation support, and Softmax is limited to the final axis
in this runtime. These constraints must be tested before
describing any imported model as runnable.

## 5. CPU execution is not universal GPU inference

KairoGPU implements selected Metal Float32 kernels; it does not
automatically execute every ONNX operator or dynamic shape. Similarly,
KairoTransformers may reuse tensor operations, but model/weight/tokenizer
semantics remain independent of the ONNX importer's supported subset.

For performance comparisons, measure graph-load/validation/optimization
time separately from warmed inference latency, allocations and native
device transfers.

## 6. Verification and reproducibility

Read [`tests/smoke.cpp`](../tests/smoke.cpp) and
[`tools/generate_reference_fixture.py`](../tools/generate_reference_fixture.py).
Compare supported graph outputs against independent reference values
with documented dtype and tolerance. Include unsupported models as
negative tests, not as silently skipped successes.

*Evidence status: source-inspection teaching artifact. No new model
compliance certification, GPU execution or benchmark was performed.*
