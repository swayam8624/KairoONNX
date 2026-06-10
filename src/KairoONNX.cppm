module;

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

export module Kairo.ONNX;

export namespace kairo::onnx
{
    enum class ElementType
    {
        Float32,
        Float64,
        Int32,
        Int64,
        UInt8,
        Bool
    };

    enum class OpKind
    {
        Unknown,
        Add,
        Sub,
        Mul,
        Div,
        Gemm,
        MatMul,
        Relu,
        Softmax,
        Conv,
        MaxPool,
        Flatten,
        Reshape,
        Transpose,
        LayerNormalization
    };

    [[nodiscard]]
    inline std::string OpKindName(OpKind op)
    {
        switch (op)
        {
        case OpKind::Add: return "Add";
        case OpKind::Sub: return "Sub";
        case OpKind::Mul: return "Mul";
        case OpKind::Div: return "Div";
        case OpKind::Gemm: return "Gemm";
        case OpKind::MatMul: return "MatMul";
        case OpKind::Relu: return "Relu";
        case OpKind::Softmax: return "Softmax";
        case OpKind::Conv: return "Conv";
        case OpKind::MaxPool: return "MaxPool";
        case OpKind::Flatten: return "Flatten";
        case OpKind::Reshape: return "Reshape";
        case OpKind::Transpose: return "Transpose";
        case OpKind::LayerNormalization: return "LayerNormalization";
        default: return "Unknown";
        }
    }

    struct TensorInfo final
    {
        std::string name;
        ElementType elementType = ElementType::Float32;
        std::vector<std::int64_t> shape;
    };

    struct Attribute final
    {
        std::string name;
        std::string value;
    };

    struct Node final
    {
        std::string name;
        OpKind op = OpKind::Unknown;
        std::vector<std::string> inputs;
        std::vector<std::string> outputs;
        std::vector<Attribute> attributes;
    };

    struct Graph final
    {
        std::vector<TensorInfo> inputs;
        std::vector<TensorInfo> outputs;
        std::vector<TensorInfo> initializers;
        std::vector<Node> nodes;

        [[nodiscard]]
        const TensorInfo* FindTensor(const std::string& name) const noexcept
        {
            for (const TensorInfo& tensor : inputs)
            {
                if (tensor.name == name) return &tensor;
            }
            for (const TensorInfo& tensor : outputs)
            {
                if (tensor.name == name) return &tensor;
            }
            for (const TensorInfo& tensor : initializers)
            {
                if (tensor.name == name) return &tensor;
            }
            return nullptr;
        }
    };

    struct ImportResult final
    {
        Graph graph;
        std::vector<std::string> diagnostics;

        [[nodiscard]]
        bool Success() const noexcept
        {
            return diagnostics.empty();
        }
    };

    [[nodiscard]]
    inline bool IsSupportedInferenceOp(OpKind op) noexcept
    {
        switch (op)
        {
        case OpKind::Add:
        case OpKind::Sub:
        case OpKind::Mul:
        case OpKind::Div:
        case OpKind::Gemm:
        case OpKind::MatMul:
        case OpKind::Relu:
        case OpKind::Softmax:
        case OpKind::Conv:
        case OpKind::MaxPool:
        case OpKind::Flatten:
        case OpKind::Reshape:
        case OpKind::Transpose:
        case OpKind::LayerNormalization:
            return true;
        default:
            return false;
        }
    }

    [[nodiscard]]
    inline std::vector<std::string> ValidateGraph(const Graph& graph)
    {
        std::vector<std::string> diagnostics;
        if (graph.inputs.empty())
        {
            diagnostics.push_back("Graph has no inputs.");
        }
        if (graph.outputs.empty())
        {
            diagnostics.push_back("Graph has no outputs.");
        }
        for (const Node& node : graph.nodes)
        {
            if (!IsSupportedInferenceOp(node.op))
            {
                diagnostics.push_back("Unsupported op: " + node.name + " (" + OpKindName(node.op) + ")");
            }
            if (node.outputs.empty())
            {
                diagnostics.push_back("Node has no outputs: " + node.name);
            }
        }
        return diagnostics;
    }

    [[nodiscard]]
    inline std::optional<TensorInfo> InferElementwiseOutput(
        const Graph& graph,
        const Node& node)
    {
        if (node.inputs.empty() || node.outputs.empty())
        {
            return std::nullopt;
        }
        const TensorInfo* input = graph.FindTensor(node.inputs[0]);
        if (!input)
        {
            return std::nullopt;
        }
        TensorInfo out = *input;
        out.name = node.outputs[0];
        return out;
    }

    [[nodiscard]]
    inline ImportResult ImportModelBytes(const std::vector<std::byte>& bytes)
    {
        ImportResult result;
        if (bytes.empty())
        {
            result.diagnostics.push_back("ONNX payload is empty.");
            return result;
        }
        result.diagnostics.push_back("ONNX protobuf parser is not implemented yet.");
        return result;
    }
}
