module;

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

export module Kairo.ONNX.Optimize;

export import Kairo.ONNX;
import Kairo.ONNX.Runtime;
import Kairo.Foundation.Math.Tensor;

export namespace kairo::onnx
{
    namespace optimize_detail
    {
        [[nodiscard]] inline const Attribute* AttributeByName(
            const Node& node, std::string_view name) noexcept
        {
            for (const Attribute& attribute : node.attributes)
                if (attribute.name == name) return &attribute;
            return nullptr;
        }

        [[nodiscard]] inline std::int64_t Integer(
            const Node& node, std::string_view name, std::int64_t fallback)
        {
            const Attribute* attribute = AttributeByName(node, name);
            return attribute ? std::stoll(attribute->value) : fallback;
        }

        [[nodiscard]] inline std::vector<std::int64_t> Integers(
            const Node& node,
            std::string_view name,
            std::vector<std::int64_t> fallback)
        {
            const Attribute* attribute = AttributeByName(node, name);
            if (!attribute) return fallback;
            std::vector<std::int64_t> result;
            std::size_t begin = 0;
            while (begin <= attribute->value.size())
            {
                const std::size_t end = attribute->value.find(',', begin);
                result.push_back(std::stoll(attribute->value.substr(
                    begin, end == std::string::npos ? end : end - begin)));
                if (end == std::string::npos) break;
                begin = end + 1;
            }
            return result;
        }

        [[nodiscard]] inline std::vector<std::int64_t> Broadcast(
            const std::vector<std::int64_t>& lhs,
            const std::vector<std::int64_t>& rhs)
        {
            const std::size_t rank = std::max(lhs.size(), rhs.size());
            std::vector<std::int64_t> output(rank, 1);
            for (std::size_t trailing = 0; trailing < rank; ++trailing)
            {
                const std::int64_t left =
                    trailing < lhs.size() ? lhs[lhs.size() - 1 - trailing] : 1;
                const std::int64_t right =
                    trailing < rhs.size() ? rhs[rhs.size() - 1 - trailing] : 1;
                if (left == -1 || right == -1)
                    output[rank - 1 - trailing] = std::max(left, right);
                else if (left != right && left != 1 && right != 1)
                    throw std::invalid_argument("Static broadcast shape conflict.");
                else
                    output[rank - 1 - trailing] = std::max(left, right);
            }
            return output;
        }

        [[nodiscard]] inline TensorInfo FloatTensorInfo(
            std::string name, const Tensor<float>& tensor)
        {
            TensorInfo info{
                .name = std::move(name),
                .elementType = ElementType::Float32,
                .shape = {},
                .rawData = std::vector<std::byte>(tensor.Size() * sizeof(float))
            };
            for (std::size_t axis = 0; axis < tensor.Rank(); ++axis)
                info.shape.push_back(static_cast<std::int64_t>(tensor.Dim(axis)));
            const Tensor<float> contiguous = tensor.Contiguous();
            std::memcpy(info.rawData.data(), contiguous.Data(), info.rawData.size());
            return info;
        }

        [[nodiscard]] inline std::vector<std::int64_t> Int64Values(
            const TensorInfo& tensor)
        {
            if (tensor.elementType != ElementType::Int64
                || tensor.rawData.size() % sizeof(std::int64_t) != 0)
                throw std::invalid_argument(
                    "Static shape input must be a valid Int64 initializer.");
            std::vector<std::int64_t> values(
                tensor.rawData.size() / sizeof(std::int64_t));
            std::memcpy(values.data(), tensor.rawData.data(), tensor.rawData.size());
            return values;
        }
    }

    /// Propagates static dtype/shape metadata in graph order and records
    /// intermediate values in `Graph::valueInfo`. Symbolic dimensions remain
    /// -1. Unsupported dynamic shape arithmetic fails explicitly.
    inline void InferStaticShapes(Graph& graph)
    {
        using namespace optimize_detail;
        std::unordered_map<std::string, TensorInfo> known;
        for (const TensorInfo& input : graph.inputs) known[input.name] = input;
        for (const TensorInfo& initializer : graph.initializers)
            known[initializer.name] = initializer;
        graph.valueInfo.clear();
        for (const Node& node : graph.nodes)
        {
            if (node.outputs.size() != 1)
                throw std::invalid_argument("Shape inference requires one output per node.");
            const auto input = [&](std::size_t index) -> const TensorInfo&
            {
                if (index >= node.inputs.size() || !known.contains(node.inputs[index]))
                    throw std::invalid_argument(
                        "Shape inference missing node input: " + node.name);
                return known.at(node.inputs[index]);
            };
            TensorInfo output{
                .name = node.outputs[0],
                .elementType = input(0).elementType
            };
            switch (node.op)
            {
            case OpKind::Add:
            case OpKind::Sub:
            case OpKind::Mul:
            case OpKind::Div:
                output.shape = Broadcast(input(0).shape, input(1).shape);
                break;
            case OpKind::Relu:
            case OpKind::Gelu:
            case OpKind::Sigmoid:
            case OpKind::Softmax:
            case OpKind::LayerNormalization:
                output.shape = input(0).shape;
                break;
            case OpKind::MatMul:
            {
                const auto& lhs = input(0).shape;
                const auto& rhs = input(1).shape;
                if (lhs.size() < 2 || rhs.size() != lhs.size()
                    || (lhs.back() != -1 && rhs[rhs.size() - 2] != -1
                        && lhs.back() != rhs[rhs.size() - 2]))
                    throw std::invalid_argument("Static MatMul shape mismatch.");
                output.shape = lhs;
                output.shape.back() = rhs.back();
                break;
            }
            case OpKind::Gemm:
            {
                const auto& lhs = input(0).shape;
                const auto& rhs = input(1).shape;
                if (lhs.size() != 2 || rhs.size() != 2)
                    throw std::invalid_argument("Static Gemm requires rank two.");
                output.shape = {
                    Integer(node, "transA", 0) ? lhs[1] : lhs[0],
                    Integer(node, "transB", 0) ? rhs[0] : rhs[1]
                };
                break;
            }
            case OpKind::Flatten:
            {
                const auto& shape = input(0).shape;
                std::int64_t axis = Integer(node, "axis", 1);
                if (axis < 0) axis += static_cast<std::int64_t>(shape.size());
                if (axis < 0 || axis > static_cast<std::int64_t>(shape.size()))
                    throw std::invalid_argument("Flatten axis out of range.");
                std::int64_t before = 1, after = 1;
                for (std::size_t index = 0; index < shape.size(); ++index)
                {
                    if (shape[index] < 0)
                    {
                        if (static_cast<std::int64_t>(index) < axis) before = -1;
                        else after = -1;
                    }
                    else if (static_cast<std::int64_t>(index) < axis && before >= 0)
                        before *= shape[index];
                    else if (static_cast<std::int64_t>(index) >= axis && after >= 0)
                        after *= shape[index];
                }
                output.shape = { before, after };
                break;
            }
            case OpKind::Reshape:
            {
                std::vector<std::int64_t> requested = Int64Values(input(1));
                if (requested.empty())
                    throw std::invalid_argument("Reshape shape cannot be empty.");
                std::int64_t inferredAxis = -1;
                std::int64_t knownProduct = 1;
                std::int64_t inputProduct = 1;
                for (std::int64_t dimension : input(0).shape)
                {
                    if (dimension < 0)
                        throw std::invalid_argument(
                            "Dynamic input product cannot infer Reshape.");
                    inputProduct *= dimension;
                }
                for (std::size_t axis = 0; axis < requested.size(); ++axis)
                {
                    if (requested[axis] == 0)
                    {
                        if (axis >= input(0).shape.size())
                            throw std::invalid_argument("Reshape zero axis is out of range.");
                        requested[axis] = input(0).shape[axis];
                    }
                    if (requested[axis] == -1)
                    {
                        if (inferredAxis >= 0)
                            throw std::invalid_argument("Reshape has multiple inferred axes.");
                        inferredAxis = static_cast<std::int64_t>(axis);
                    }
                    else if (requested[axis] <= 0)
                        throw std::invalid_argument("Reshape dimension is invalid.");
                    else
                        knownProduct *= requested[axis];
                }
                if (inferredAxis >= 0)
                {
                    if (knownProduct == 0 || inputProduct % knownProduct != 0)
                        throw std::invalid_argument("Reshape product is incompatible.");
                    requested[static_cast<std::size_t>(inferredAxis)] =
                        inputProduct / knownProduct;
                }
                else if (knownProduct != inputProduct)
                    throw std::invalid_argument("Reshape product is incompatible.");
                output.shape = std::move(requested);
                break;
            }
            case OpKind::Gather:
            {
                const auto& dataShape = input(0).shape;
                const auto& indexShape = input(1).shape;
                std::int64_t axis = Integer(node, "axis", 0);
                if (axis < 0) axis += static_cast<std::int64_t>(dataShape.size());
                if (axis < 0 || axis >= static_cast<std::int64_t>(dataShape.size()))
                    throw std::invalid_argument("Gather axis is out of range.");
                output.shape.insert(
                    output.shape.end(), dataShape.begin(), dataShape.begin() + axis);
                output.shape.insert(
                    output.shape.end(), indexShape.begin(), indexShape.end());
                output.shape.insert(
                    output.shape.end(), dataShape.begin() + axis + 1, dataShape.end());
                break;
            }
            case OpKind::Slice:
            {
                output.shape = input(0).shape;
                const auto starts = Int64Values(input(1));
                const auto ends = Int64Values(input(2));
                const auto axes = node.inputs.size() > 3
                    ? Int64Values(input(3))
                    : std::vector<std::int64_t>{};
                const auto steps = node.inputs.size() > 4
                    ? Int64Values(input(4))
                    : std::vector<std::int64_t>{};
                if (starts.size() != ends.size()
                    || (!axes.empty() && axes.size() != starts.size())
                    || (!steps.empty() && steps.size() != starts.size()))
                    throw std::invalid_argument("Slice static inputs have unequal lengths.");
                for (std::size_t index = 0; index < starts.size(); ++index)
                {
                    std::int64_t axis = axes.empty()
                        ? static_cast<std::int64_t>(index) : axes[index];
                    if (axis < 0) axis += static_cast<std::int64_t>(output.shape.size());
                    const std::int64_t step = steps.empty() ? 1 : steps[index];
                    if (axis < 0 || axis >= static_cast<std::int64_t>(output.shape.size())
                        || step <= 0 || output.shape[static_cast<std::size_t>(axis)] < 0)
                        throw std::invalid_argument("Slice static axis/step is unsupported.");
                    const std::int64_t dimension =
                        output.shape[static_cast<std::size_t>(axis)];
                    const std::int64_t begin = std::clamp(starts[index], -dimension, dimension);
                    const std::int64_t end = std::clamp(ends[index], -dimension, dimension);
                    const std::int64_t normalizedBegin =
                        begin < 0 ? begin + dimension : begin;
                    const std::int64_t normalizedEnd =
                        end < 0 ? end + dimension : end;
                    output.shape[static_cast<std::size_t>(axis)] =
                        std::max<std::int64_t>(
                            0, (normalizedEnd - normalizedBegin + step - 1) / step);
                }
                break;
            }
            case OpKind::Concat:
            {
                output.shape = input(0).shape;
                std::int64_t axis = Integer(node, "axis", 0);
                if (axis < 0) axis += static_cast<std::int64_t>(output.shape.size());
                if (axis < 0 || axis >= static_cast<std::int64_t>(output.shape.size()))
                    throw std::invalid_argument("Concat axis is out of range.");
                std::int64_t extent = 0;
                for (std::size_t index = 0; index < node.inputs.size(); ++index)
                {
                    const auto& shape = input(index).shape;
                    if (shape.size() != output.shape.size())
                        throw std::invalid_argument("Concat rank mismatch.");
                    for (std::size_t dimension = 0; dimension < shape.size(); ++dimension)
                        if (dimension != static_cast<std::size_t>(axis)
                            && shape[dimension] != output.shape[dimension])
                            throw std::invalid_argument("Concat non-axis shape mismatch.");
                    if (shape[static_cast<std::size_t>(axis)] < 0) extent = -1;
                    else if (extent >= 0) extent += shape[static_cast<std::size_t>(axis)];
                }
                output.shape[static_cast<std::size_t>(axis)] = extent;
                break;
            }
            case OpKind::Transpose:
            {
                const auto& shape = input(0).shape;
                const auto permutation = Integers(node, "perm", {});
                if (permutation.empty())
                    output.shape.assign(shape.rbegin(), shape.rend());
                else
                {
                    if (permutation.size() != shape.size())
                        throw std::invalid_argument("Transpose permutation rank mismatch.");
                    output.shape.resize(shape.size());
                    std::unordered_set<std::int64_t> used;
                    for (std::size_t index = 0; index < permutation.size(); ++index)
                    {
                        if (permutation[index] < 0
                            || permutation[index] >= static_cast<std::int64_t>(shape.size())
                            || !used.emplace(permutation[index]).second)
                            throw std::invalid_argument("Invalid transpose permutation.");
                        output.shape[index] = shape[static_cast<std::size_t>(permutation[index])];
                    }
                }
                break;
            }
            case OpKind::Conv:
            {
                const auto& data = input(0).shape;
                const auto& weight = input(1).shape;
                const auto strides = Integers(node, "strides", { 1, 1 });
                const auto pads = Integers(node, "pads", { 0, 0, 0, 0 });
                if (data.size() != 4 || weight.size() != 4
                    || strides.size() != 2 || pads.size() != 4)
                    throw std::invalid_argument("Static Conv requires NCHW/OIHW 2D shapes.");
                output.shape = {
                    data[0], weight[0],
                    data[2] < 0 || weight[2] < 0 ? -1
                        : (data[2] + pads[0] + pads[2] - weight[2]) / strides[0] + 1,
                    data[3] < 0 || weight[3] < 0 ? -1
                        : (data[3] + pads[1] + pads[3] - weight[3]) / strides[1] + 1
                };
                break;
            }
            case OpKind::MaxPool:
            {
                const auto& data = input(0).shape;
                const auto kernel = Integers(node, "kernel_shape", {});
                const auto strides = Integers(node, "strides", { 1, 1 });
                const auto pads = Integers(node, "pads", { 0, 0, 0, 0 });
                if (data.size() != 4 || kernel.size() != 2)
                    throw std::invalid_argument("Static MaxPool requires rank-4 and 2D kernel.");
                output.shape = {
                    data[0], data[1],
                    data[2] < 0 ? -1
                        : (data[2] + pads[0] + pads[2] - kernel[0]) / strides[0] + 1,
                    data[3] < 0 ? -1
                        : (data[3] + pads[1] + pads[3] - kernel[1]) / strides[1] + 1
                };
                break;
            }
            default:
                throw std::invalid_argument(
                    "Static shape inference not implemented for " + OpKindName(node.op));
            }
            known[output.name] = output;
            graph.valueInfo.push_back(output);
        }
        for (TensorInfo& output : graph.outputs)
            if (known.contains(output.name))
            {
                output.elementType = known.at(output.name).elementType;
                output.shape = known.at(output.name).shape;
            }
    }

    /// Evaluates nodes whose complete inputs are initializers. Folded outputs
    /// become Float32 initializers and no longer execute at runtime.
    inline void ConstantFold(Graph& graph)
    {
        using namespace optimize_detail;
        std::unordered_map<std::string, TensorInfo> constants;
        for (const TensorInfo& initializer : graph.initializers)
            constants[initializer.name] = initializer;
        std::vector<Node> retained;
        for (const Node& node : graph.nodes)
        {
            bool foldable = !node.inputs.empty();
            for (const std::string& input : node.inputs)
                foldable = foldable && !input.empty() && constants.contains(input);
            if (!foldable)
            {
                retained.push_back(node);
                continue;
            }
            Graph subgraph;
            subgraph.inputs.push_back(constants.at(node.inputs[0]));
            subgraph.outputs.push_back({ .name = node.outputs.at(0) });
            subgraph.nodes.push_back(node);
            std::unordered_set<std::string> inserted;
            for (const std::string& input : node.inputs)
                if (inserted.emplace(input).second)
                    subgraph.initializers.push_back(constants.at(input));
            const ExecutionResult execution = ExecuteGraph(subgraph, {});
            const auto* tensor =
                std::get_if<Tensor<float>>(&execution.outputs.at(node.outputs[0]));
            if (!tensor)
                throw std::invalid_argument("Constant folding produced a non-Float32 value.");
            TensorInfo folded = FloatTensorInfo(node.outputs[0], *tensor);
            constants[folded.name] = folded;
            graph.initializers.push_back(std::move(folded));
        }
        graph.nodes = std::move(retained);
    }

    /// Removes nodes and intermediate constants that cannot reach a declared
    /// graph output. Original topological order is preserved.
    inline void EliminateDeadValues(Graph& graph)
    {
        std::unordered_set<std::string> required;
        for (const TensorInfo& output : graph.outputs) required.emplace(output.name);
        std::vector<Node> retained;
        for (auto iterator = graph.nodes.rbegin(); iterator != graph.nodes.rend(); ++iterator)
        {
            bool live = false;
            for (const std::string& output : iterator->outputs)
                live = live || required.contains(output);
            if (!live) continue;
            retained.push_back(*iterator);
            for (const std::string& input : iterator->inputs)
                if (!input.empty()) required.emplace(input);
        }
        std::reverse(retained.begin(), retained.end());
        graph.nodes = std::move(retained);
        std::erase_if(graph.initializers, [&](const TensorInfo& initializer)
        {
            return !required.contains(initializer.name);
        });
        std::erase_if(graph.valueInfo, [&](const TensorInfo& value)
        {
            return !required.contains(value.name);
        });
    }
}
