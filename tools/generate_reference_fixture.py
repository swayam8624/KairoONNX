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


def save_reference(name, graph, input_name, input_value, output_name):
    fixture_model = helper.make_model(
        graph,
        producer_name="KairoONNX reference fixture",
        opset_imports=[helper.make_opsetid("", 18)],
    )
    fixture_model.ir_version = 10
    onnx.checker.check_model(fixture_model)
    fixture_path = FIXTURES / f"reference_{name}.onnx"
    onnx.save(fixture_model, fixture_path)
    fixture_session = ort.InferenceSession(
        fixture_path.read_bytes(), providers=["CPUExecutionProvider"]
    )
    fixture_output = fixture_session.run(
        [output_name], {input_name: input_value}
    )[0].astype(np.float32)
    (FIXTURES / f"reference_{name}_input.f32").write_bytes(input_value.tobytes())
    (FIXTURES / f"reference_{name}_output.f32").write_bytes(fixture_output.tobytes())
    print(name, fixture_output)


image = np.arange(1, 10, dtype=np.float32).reshape(1, 1, 3, 3)
cnn_graph = helper.make_graph(
    [
        helper.make_node("Conv", ["image", "kernel", "conv_bias"], ["conv"]),
        helper.make_node(
            "MaxPool", ["conv"], ["pooled"], kernel_shape=[2, 2], strides=[2, 2]
        ),
    ],
    "kairo_reference_cnn",
    [helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 1, 3, 3])],
    [helper.make_tensor_value_info("pooled", TensorProto.FLOAT, [1, 1, 1, 1])],
    [
        numpy_helper.from_array(
            np.array([[[[1.0, 0.0], [0.0, -1.0]]]], dtype=np.float32), "kernel"
        ),
        numpy_helper.from_array(np.array([0.25], dtype=np.float32), "conv_bias"),
    ],
)
save_reference("cnn", cnn_graph, "image", image, "pooled")

hidden = np.array(
    [[0.2, -0.1, 0.4, 0.7], [-0.3, 0.8, 0.1, -0.2]], dtype=np.float32
)
identity = np.eye(4, dtype=np.float32)
transformer_graph = helper.make_graph(
    [
        helper.make_node(
            "LayerNormalization",
            ["hidden", "norm_scale", "norm_bias"],
            ["normalized"],
            axis=-1,
            epsilon=1e-5,
        ),
        helper.make_node("MatMul", ["normalized", "q_weight"], ["query"]),
        helper.make_node("MatMul", ["normalized", "k_weight"], ["key"]),
        helper.make_node("MatMul", ["normalized", "v_weight"], ["value"]),
        helper.make_node("Transpose", ["key"], ["key_t"], perm=[1, 0]),
        helper.make_node("MatMul", ["query", "key_t"], ["scores"]),
        helper.make_node("Div", ["scores", "attention_scale"], ["scaled"]),
        helper.make_node("Softmax", ["scaled"], ["probabilities"], axis=-1),
        helper.make_node("MatMul", ["probabilities", "value"], ["attended"]),
    ],
    "kairo_reference_transformer_attention",
    [helper.make_tensor_value_info("hidden", TensorProto.FLOAT, [2, 4])],
    [helper.make_tensor_value_info("attended", TensorProto.FLOAT, [2, 4])],
    [
        numpy_helper.from_array(np.ones(4, dtype=np.float32), "norm_scale"),
        numpy_helper.from_array(np.zeros(4, dtype=np.float32), "norm_bias"),
        numpy_helper.from_array(identity, "q_weight"),
        numpy_helper.from_array(identity * 0.75, "k_weight"),
        numpy_helper.from_array(identity * 1.25, "v_weight"),
        numpy_helper.from_array(np.array([2.0], dtype=np.float32), "attention_scale"),
    ],
)
save_reference(
    "transformer", transformer_graph, "hidden", hidden, "attended"
)
