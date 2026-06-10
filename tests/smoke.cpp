import Kairo.ONNX;

#include <cassert>

int main()
{
    assert(kairo::onnx::IsSupportedInferenceOp(kairo::onnx::OpKind::MatMul));
    assert(!kairo::onnx::IsSupportedInferenceOp(kairo::onnx::OpKind::Unknown));
    kairo::onnx::Graph graph;
    graph.inputs.push_back({ .name = "x", .shape = { 1, 4 } });
    graph.outputs.push_back({ .name = "y", .shape = { 1, 4 } });
    graph.nodes.push_back({ .name = "relu0", .op = kairo::onnx::OpKind::Relu, .inputs = { "x" }, .outputs = { "y" } });
    assert(kairo::onnx::ValidateGraph(graph).empty());
    return 0;
}
