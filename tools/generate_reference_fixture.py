"""Generate the bounded MLP conformance fixture with a trusted ONNX Runtime output."""

from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
FIXTURES.mkdir(parents=True, exist_ok=True)

features = np.array([[1.0, 2.0, 3.0], [-2.0, 1.0, 0.5]], dtype=np.float32)
weight = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, -1.0]], dtype=np.float32)
bias = np.array([0.5, -0.5], dtype=np.float32)

graph = helper.make_graph(
    [
        helper.make_node("MatMul", ["features", "weight"], ["dense"]),
        helper.make_node("Add", ["dense", "bias"], ["biased"]),
        helper.make_node("Relu", ["biased"], ["activated"]),
    ],
    "kairo_reference_mlp",
    [helper.make_tensor_value_info("features", TensorProto.FLOAT, [2, 3])],
    [helper.make_tensor_value_info("activated", TensorProto.FLOAT, [2, 2])],
    [numpy_helper.from_array(weight, "weight"), numpy_helper.from_array(bias, "bias")],
)
model = helper.make_model(
    graph,
    producer_name="KairoONNX reference fixture",
    opset_imports=[helper.make_opsetid("", 18)],
)
model.ir_version = 10
onnx.checker.check_model(model)
model_path = FIXTURES / "reference_mlp.onnx"
onnx.save(model, model_path)

session = ort.InferenceSession(model_path.read_bytes(), providers=["CPUExecutionProvider"])
output = session.run(["activated"], {"features": features})[0].astype(np.float32)
(FIXTURES / "reference_mlp_input.f32").write_bytes(features.tobytes())
(FIXTURES / "reference_mlp_output.f32").write_bytes(output.tobytes())
print(output)
