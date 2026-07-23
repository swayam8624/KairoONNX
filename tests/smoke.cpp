import Kairo.ONNX;

#include <cassert>
#include <cstddef>
#include <vector>

int main()
{
    assert(kairo::onnx::IsSupportedInferenceOp(kairo::onnx::OpKind::MatMul));
    assert(!kairo::onnx::IsSupportedInferenceOp(kairo::onnx::OpKind::Unknown));
    kairo::onnx::Graph graph;
    graph.inputs.push_back({ .name = "x", .shape = { 1, 4 } });
    graph.outputs.push_back({ .name = "y", .shape = { 1, 4 } });
    graph.nodes.push_back({ .name = "relu0", .op = kairo::onnx::OpKind::Relu, .inputs = { "x" }, .outputs = { "y" } });
    assert(kairo::onnx::ValidateGraph(graph).empty());

    // Minimal protobuf ModelProto: graph { node { input: "x" output: "y"
    // op_type: "Relu" } input { name: "x" } output { name: "y" } }.
    const std::vector<std::byte> modelBytes = {
        std::byte{0x3A}, std::byte{0x35},
        std::byte{0x0A}, std::byte{0x0C}, std::byte{0x0A}, std::byte{0x01}, std::byte{0x78},
        std::byte{0x12}, std::byte{0x01}, std::byte{0x79}, std::byte{0x22}, std::byte{0x04},
        std::byte{0x52}, std::byte{0x65}, std::byte{0x6C}, std::byte{0x75},
        std::byte{0x5A}, std::byte{0x03}, std::byte{0x0A}, std::byte{0x01}, std::byte{0x78},
        std::byte{0x62}, std::byte{0x03}, std::byte{0x0A}, std::byte{0x01}, std::byte{0x79},
        std::byte{0x2A}, std::byte{0x1B},
        std::byte{0x08}, std::byte{0x02}, std::byte{0x08}, std::byte{0x02},
        std::byte{0x10}, std::byte{0x01}, std::byte{0x42}, std::byte{0x01}, std::byte{0x77},
        std::byte{0x4A}, std::byte{0x10},
        std::byte{0x00}, std::byte{0x00}, std::byte{0x80}, std::byte{0x3F},
        std::byte{0x00}, std::byte{0x00}, std::byte{0x00}, std::byte{0x40},
        std::byte{0x00}, std::byte{0x00}, std::byte{0x40}, std::byte{0x40},
        std::byte{0x00}, std::byte{0x00}, std::byte{0x80}, std::byte{0x40},
    };
    const auto imported = kairo::onnx::ImportModelBytes(modelBytes);
    assert(imported.Success());
    assert(imported.graph.nodes.size() == 1);
    assert(imported.graph.nodes[0].op == kairo::onnx::OpKind::Relu);
    assert(imported.graph.inputs[0].name == "x");
    assert(imported.graph.outputs[0].name == "y");
    assert(imported.graph.initializers.size() == 1);
    assert(imported.graph.initializers[0].name == "w");
    assert((imported.graph.initializers[0].shape == std::vector<std::int64_t>{ 2, 2 }));
    assert(imported.graph.initializers[0].rawData.size() == 16);
    return 0;
}
