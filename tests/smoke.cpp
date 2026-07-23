#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

import Kairo.ONNX;
import Kairo.ONNX.Runtime;
import Kairo.ONNX.Optimize;
import Kairo.Foundation.Math.Tensor;

namespace
{
    std::vector<std::byte> ReadBytes(const std::filesystem::path& path)
    {
        std::ifstream stream(path, std::ios::binary | std::ios::ate);
        if (!stream) throw std::runtime_error("Cannot open ONNX conformance fixture.");
        const std::streamsize size = stream.tellg();
        if (size <= 0) throw std::runtime_error("ONNX conformance fixture is empty.");
        stream.seekg(0);
        std::vector<std::byte> bytes(static_cast<std::size_t>(size));
        if (!stream.read(reinterpret_cast<char*>(bytes.data()), size))
            throw std::runtime_error("Cannot read ONNX conformance fixture.");
        return bytes;
    }

    kairo::onnx::TensorInfo Initializer(
        std::string name, std::vector<std::int64_t> shape, std::vector<float> values)
    {
        kairo::onnx::TensorInfo result{
            .name = std::move(name),
            .elementType = kairo::onnx::ElementType::Float32,
            .shape = std::move(shape),
            .rawData = std::vector<std::byte>(values.size() * sizeof(float))
        };
        std::memcpy(result.rawData.data(), values.data(), result.rawData.size());
        return result;
    }

    kairo::onnx::TensorInfo IntInitializer(
        std::string name, std::vector<std::int64_t> values)
    {
        kairo::onnx::TensorInfo result{
            .name = std::move(name),
            .elementType = kairo::onnx::ElementType::Int64,
            .shape = { static_cast<std::int64_t>(values.size()) },
            .rawData = std::vector<std::byte>(values.size() * sizeof(std::int64_t))
        };
        std::memcpy(result.rawData.data(), values.data(), result.rawData.size());
        return result;
    }
}

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
        std::byte{0x42}, std::byte{0x02}, std::byte{0x10}, std::byte{0x12},
    };
    const auto imported = kairo::onnx::ImportModelBytes(modelBytes);
    assert(imported.Success());
    assert(imported.graph.nodes.size() == 1);
    assert(imported.graph.nodes[0].op == kairo::onnx::OpKind::Relu);
    assert(imported.graph.inputs[0].name == "x");
    assert(imported.graph.outputs[0].name == "y");
    assert(imported.graph.initializers.size() == 1);
    assert(imported.graph.operatorSets.size() == 1);
    assert(imported.graph.operatorSets[0].version == 18);
    assert(imported.graph.initializers[0].name == "w");
    assert((imported.graph.initializers[0].shape == std::vector<std::int64_t>{ 2, 2 }));
    assert(imported.graph.initializers[0].rawData.size() == 16);
    const auto weights = kairo::onnx::Float32InitializerTensor(imported.graph.initializers[0]);
    assert(weights.Dim(0) == 2 && weights.Dim(1) == 2);
    assert(weights(0, 0) == 1.0f && weights(1, 1) == 4.0f);

    kairo::onnx::Graph mlp;
    mlp.inputs.push_back({ .name = "features", .shape = { 2, 3 } });
    mlp.outputs.push_back({ .name = "activated", .shape = { 2, 2 } });
    mlp.initializers.push_back(Initializer("weight", { 3, 2 }, {
        1, 0, 0, 1, 1, -1
    }));
    mlp.initializers.push_back(Initializer("bias", { 2 }, { 0.5f, -0.5f }));
    mlp.nodes.push_back({
        .name = "dense",
        .op = kairo::onnx::OpKind::MatMul,
        .inputs = { "features", "weight" },
        .outputs = { "dense_out" }
    });
    mlp.nodes.push_back({
        .name = "bias_add",
        .op = kairo::onnx::OpKind::Add,
        .inputs = { "dense_out", "bias" },
        .outputs = { "biased" }
    });
    mlp.nodes.push_back({
        .name = "activation",
        .op = kairo::onnx::OpKind::Relu,
        .inputs = { "biased" },
        .outputs = { "activated" }
    });
    kairo::onnx::RuntimeBindings mlpFeeds;
    mlpFeeds.emplace("features", kairo::foundation::math::Tensor<float>(
        { 2, 3 }, { 1, 2, 3, -2, 1, 0.5f }));
    const auto mlpResult = kairo::onnx::ExecuteGraph(mlp, mlpFeeds);
    const auto& mlpOutput = std::get<kairo::foundation::math::Tensor<float>>(
        mlpResult.outputs.at("activated"));
    assert(mlpOutput(0, 0) == 4.5f && mlpOutput(0, 1) == 0.0f);
    assert(mlpOutput(1, 0) == 0.0f && mlpOutput(1, 1) == 0.0f);

    kairo::onnx::Graph cnn;
    cnn.inputs.push_back({ .name = "image", .shape = { 1, 1, 3, 3 } });
    cnn.outputs.push_back({ .name = "pooled", .shape = { 1, 1, 1, 1 } });
    cnn.initializers.push_back(Initializer("kernel", { 1, 1, 2, 2 }, {
        1, 0, 0, -1
    }));
    cnn.initializers.push_back(Initializer("conv_bias", { 1 }, { 0.25f }));
    cnn.nodes.push_back({
        .name = "conv",
        .op = kairo::onnx::OpKind::Conv,
        .inputs = { "image", "kernel", "conv_bias" },
        .outputs = { "conv_out" },
        .attributes = { { "strides", "1,1" }, { "pads", "0,0,0,0" } }
    });
    cnn.nodes.push_back({
        .name = "pool",
        .op = kairo::onnx::OpKind::MaxPool,
        .inputs = { "conv_out" },
        .outputs = { "pooled" },
        .attributes = { { "kernel_shape", "2,2" }, { "strides", "2,2" } }
    });
    kairo::onnx::RuntimeBindings cnnFeeds;
    cnnFeeds.emplace("image", kairo::foundation::math::Tensor<float>(
        { 1, 1, 3, 3 }, { 1, 2, 3, 4, 5, 6, 7, 8, 9 }));
    const auto cnnResult = kairo::onnx::ExecuteGraph(cnn, cnnFeeds);
    const auto& cnnOutput = std::get<kairo::foundation::math::Tensor<float>>(
        cnnResult.outputs.at("pooled"));
    assert(cnnOutput.At({ 0, 0, 0, 0 }) == -3.75f);

    kairo::onnx::Graph indexing;
    indexing.inputs.push_back({ .name = "table", .shape = { 3, 2 } });
    indexing.outputs.push_back({ .name = "joined", .shape = { 2, 2 } });
    indexing.initializers.push_back(IntInitializer("indices", { 2, 0 }));
    indexing.initializers.push_back(IntInitializer("starts", { 0 }));
    indexing.initializers.push_back(IntInitializer("ends", { 1 }));
    indexing.initializers.push_back(IntInitializer("axes", { 1 }));
    indexing.nodes.push_back({
        .name = "gather",
        .op = kairo::onnx::OpKind::Gather,
        .inputs = { "table", "indices" },
        .outputs = { "selected" },
        .attributes = { { "axis", "0" } }
    });
    indexing.nodes.push_back({
        .name = "slice",
        .op = kairo::onnx::OpKind::Slice,
        .inputs = { "selected", "starts", "ends", "axes" },
        .outputs = { "column" }
    });
    indexing.nodes.push_back({
        .name = "concat",
        .op = kairo::onnx::OpKind::Concat,
        .inputs = { "column", "column" },
        .outputs = { "joined" },
        .attributes = { { "axis", "1" } }
    });
    kairo::onnx::InferStaticShapes(indexing);
    assert((indexing.outputs[0].shape == std::vector<std::int64_t>{ 2, 2 }));
    kairo::onnx::RuntimeBindings indexingFeeds;
    indexingFeeds.emplace("table", kairo::foundation::math::Tensor<float>(
        { 3, 2 }, { 1, 2, 3, 4, 5, 6 }));
    const auto indexingResult = kairo::onnx::ExecuteGraph(indexing, indexingFeeds);
    const auto& joined = std::get<kairo::foundation::math::Tensor<float>>(
        indexingResult.outputs.at("joined"));
    assert(joined(0, 0) == 5.0f && joined(0, 1) == 5.0f);
    assert(joined(1, 0) == 1.0f && joined(1, 1) == 1.0f);

    kairo::onnx::Graph optimized;
    optimized.inputs.push_back({ .name = "runtime_x", .shape = { 1, 2 } });
    optimized.outputs.push_back({ .name = "runtime_y" });
    optimized.initializers.push_back(Initializer("constant_a", { 1, 2 }, { 1, 2 }));
    optimized.initializers.push_back(Initializer("constant_b", { 2 }, { 3, 4 }));
    optimized.nodes.push_back({
        .name = "fold_add",
        .op = kairo::onnx::OpKind::Add,
        .inputs = { "constant_a", "constant_b" },
        .outputs = { "folded" }
    });
    optimized.nodes.push_back({
        .name = "live_add",
        .op = kairo::onnx::OpKind::Add,
        .inputs = { "runtime_x", "folded" },
        .outputs = { "runtime_y" }
    });
    optimized.nodes.push_back({
        .name = "dead_relu",
        .op = kairo::onnx::OpKind::Relu,
        .inputs = { "constant_b" },
        .outputs = { "unused" }
    });
    kairo::onnx::InferStaticShapes(optimized);
    assert((optimized.outputs[0].shape == std::vector<std::int64_t>{ 1, 2 }));
    assert(optimized.valueInfo.size() == 3);
    kairo::onnx::ConstantFold(optimized);
    assert(optimized.nodes.size() == 1);
    kairo::onnx::EliminateDeadValues(optimized);
    assert(optimized.nodes.size() == 1);
    assert(optimized.initializers.size() == 1);
    kairo::onnx::RuntimeBindings optimizedFeeds;
    optimizedFeeds.emplace("runtime_x", kairo::foundation::math::Tensor<float>(
        { 1, 2 }, { 10, 20 }));
    const auto optimizedResult = kairo::onnx::ExecuteGraph(optimized, optimizedFeeds);
    const auto& optimizedOutput = std::get<kairo::foundation::math::Tensor<float>>(
        optimizedResult.outputs.at("runtime_y"));
    assert(optimizedOutput(0, 0) == 14.0f && optimizedOutput(0, 1) == 26.0f);

    // Generated by tools/generate_reference_fixture.py and evaluated by the
    // pinned ONNX Runtime tool environment before being checked in.
    const std::filesystem::path fixtures =
        std::filesystem::path(KAIRO_ONNX_SOURCE_DIR) / "tests" / "fixtures";
    const auto referenceImport =
        kairo::onnx::ImportModelBytes(ReadBytes(fixtures / "reference_mlp.onnx"));
    assert(referenceImport.Success());
    const std::vector<std::byte> referenceInput =
        ReadBytes(fixtures / "reference_mlp_input.f32");
    const std::vector<std::byte> referenceExpected =
        ReadBytes(fixtures / "reference_mlp_output.f32");
    assert(referenceInput.size() == 6 * sizeof(float));
    assert(referenceExpected.size() == 4 * sizeof(float));
    kairo::foundation::math::Tensor<float> referenceFeatures({ 2, 3 });
    std::memcpy(referenceFeatures.Data(), referenceInput.data(), referenceInput.size());
    kairo::onnx::RuntimeBindings referenceFeeds;
    referenceFeeds.emplace("features", std::move(referenceFeatures));
    const auto referenceResult =
        kairo::onnx::ExecuteGraph(referenceImport.graph, referenceFeeds);
    const auto& referenceActual =
        std::get<kairo::foundation::math::Tensor<float>>(
            referenceResult.outputs.at("activated"));
    std::vector<float> expectedValues(4);
    std::memcpy(
        expectedValues.data(), referenceExpected.data(), referenceExpected.size());
    for (std::size_t index = 0; index < expectedValues.size(); ++index)
        assert(std::abs(referenceActual[index] - expectedValues[index]) <= 1e-6f);

    bool missingRejected = false;
    try
    {
        (void)kairo::onnx::ExecuteGraph(mlp, {});
    }
    catch (const std::invalid_argument&)
    {
        missingRejected = true;
    }
    assert(missingRejected);
    return 0;
}
