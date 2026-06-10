module;

#include <cstddef>
#include <cstdint>
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
