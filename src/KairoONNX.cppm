module;

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <bit>
#include <cstring>
#include <optional>
#include <limits>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

export module Kairo.ONNX;

import Kairo.Foundation.Math.Tensor;

export namespace kairo::onnx
{
    using kairo::foundation::math::Tensor;
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
        Gelu,
        Sigmoid,
        Softmax,
        Conv,
        MaxPool,
        Flatten,
        Reshape,
        Transpose,
        LayerNormalization,
        Gather,
        Slice,
        Concat
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
        case OpKind::Gelu: return "Gelu";
        case OpKind::Sigmoid: return "Sigmoid";
        case OpKind::Softmax: return "Softmax";
        case OpKind::Conv: return "Conv";
        case OpKind::MaxPool: return "MaxPool";
        case OpKind::Flatten: return "Flatten";
        case OpKind::Reshape: return "Reshape";
        case OpKind::Transpose: return "Transpose";
        case OpKind::LayerNormalization: return "LayerNormalization";
        case OpKind::Gather: return "Gather";
        case OpKind::Slice: return "Slice";
        case OpKind::Concat: return "Concat";
        default: return "Unknown";
        }
    }

    struct TensorInfo final
    {
        std::string name;
        ElementType elementType = ElementType::Float32;
        std::vector<std::int64_t> shape;
        std::vector<std::byte> rawData;
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

    struct OperatorSet final
    {
        std::string domain;
        std::int64_t version = 0;
    };

    struct Graph final
    {
        std::vector<TensorInfo> inputs;
        std::vector<TensorInfo> outputs;
        std::vector<TensorInfo> initializers;
        std::vector<TensorInfo> valueInfo;
        std::vector<Node> nodes;
        std::vector<OperatorSet> operatorSets;

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
            for (const TensorInfo& tensor : valueInfo)
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
        case OpKind::Gelu:
        case OpKind::Sigmoid:
        case OpKind::Softmax:
        case OpKind::Conv:
        case OpKind::MaxPool:
        case OpKind::Flatten:
        case OpKind::Reshape:
        case OpKind::Transpose:
        case OpKind::LayerNormalization:
        case OpKind::Gather:
        case OpKind::Slice:
        case OpKind::Concat:
            return true;
        default:
            return false;
        }
    }

    [[nodiscard]]
    inline std::vector<std::string> ValidateGraph(const Graph& graph)
    {
        std::vector<std::string> diagnostics;
        std::vector<std::string> knownValues;
        const auto registerValue = [&](const std::string& name, std::string_view source)
        {
            if (name.empty())
            {
                diagnostics.push_back(std::string(source) + " has an empty value name.");
                return;
            }
            if (std::find(knownValues.begin(), knownValues.end(), name) != knownValues.end())
            {
                diagnostics.push_back("Duplicate ONNX value name: " + name);
                return;
            }
            knownValues.push_back(name);
        };
        if (graph.inputs.empty())
        {
            diagnostics.push_back("Graph has no inputs.");
        }
        if (graph.outputs.empty())
        {
            diagnostics.push_back("Graph has no outputs.");
        }
        for (const OperatorSet& opset : graph.operatorSets)
        {
            if ((!opset.domain.empty() && opset.domain != "ai.onnx")
                || opset.version < 7 || opset.version > 21)
                diagnostics.push_back(
                    "Unsupported ONNX opset: "
                    + (opset.domain.empty() ? std::string("ai.onnx") : opset.domain)
                    + " version " + std::to_string(opset.version));
        }
        for (const TensorInfo& input : graph.inputs) registerValue(input.name, "Graph input");
        for (const TensorInfo& initializer : graph.initializers)
        {
            // ONNX commonly repeats initializer names in graph inputs.
            if (std::find(knownValues.begin(), knownValues.end(), initializer.name)
                == knownValues.end())
                registerValue(initializer.name, "Initializer");
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
            for (const std::string& input : node.inputs)
                if (!input.empty()
                    && std::find(knownValues.begin(), knownValues.end(), input) == knownValues.end())
                    diagnostics.push_back(
                        "Node input is not topologically available: " + node.name + " <- " + input);
            for (const std::string& output : node.outputs)
                registerValue(output, "Node output");
        }
        for (const TensorInfo& output : graph.outputs)
            if (std::find(knownValues.begin(), knownValues.end(), output.name) == knownValues.end())
                diagnostics.push_back("Graph output has no producer: " + output.name);
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

    /// Converts a validated little-endian ONNX Float32 raw initializer into an
    /// owned Kairo tensor. Typed-field TensorProto encodings and non-Float32
    /// values are intentionally rejected until their conversion rules exist.
    [[nodiscard]]
    inline Tensor<float> Float32InitializerTensor(const TensorInfo& initializer)
    {
        if (initializer.elementType != ElementType::Float32)
        {
            throw std::invalid_argument("Float32InitializerTensor requires an ONNX Float32 initializer.");
        }
        if constexpr (std::endian::native != std::endian::little)
        {
            throw std::runtime_error("Float32 ONNX raw_data conversion requires a little-endian host.");
        }
        std::vector<std::size_t> shape;
        std::size_t values = 1;
        for (std::int64_t dimension : initializer.shape)
        {
            if (dimension <= 0 || static_cast<std::uint64_t>(dimension) > std::numeric_limits<std::size_t>::max() / values)
            {
                throw std::invalid_argument("ONNX initializer has an invalid or overflowing shape.");
            }
            values *= static_cast<std::size_t>(dimension);
            shape.push_back(static_cast<std::size_t>(dimension));
        }
        if (shape.empty() || initializer.rawData.size() != values * sizeof(float))
        {
            throw std::invalid_argument("ONNX Float32 raw_data length does not match initializer shape.");
        }
        std::vector<float> data(values);
        std::memcpy(data.data(), initializer.rawData.data(), initializer.rawData.size());
        return Tensor<float>(std::move(shape), std::move(data));
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
        struct Reader final
        {
            const std::vector<std::byte>& bytes;
            std::size_t position = 0;

            [[nodiscard]] bool ReadVarint(std::uint64_t& value)
            {
                value = 0;
                for (std::size_t shift = 0; shift < 64; shift += 7)
                {
                    if (position >= bytes.size()) return false;
                    const std::uint8_t byte = std::to_integer<std::uint8_t>(bytes[position++]);
                    value |= static_cast<std::uint64_t>(byte & 0x7F) << shift;
                    if ((byte & 0x80) == 0) return true;
                }
                return false;
            }

            [[nodiscard]] bool ReadField(std::uint32_t& field, std::uint32_t& wire)
            {
                std::uint64_t tag = 0;
                if (!ReadVarint(tag) || tag == 0) return false;
                field = static_cast<std::uint32_t>(tag >> 3);
                wire = static_cast<std::uint32_t>(tag & 7);
                return true;
            }

            [[nodiscard]] bool ReadFixed32(std::uint32_t& value)
            {
                if (bytes.size() - position < sizeof(value)) return false;
                std::memcpy(&value, bytes.data() + position, sizeof(value));
                position += sizeof(value);
                return true;
            }

            [[nodiscard]] bool ReadBytes(std::vector<std::byte>& out)
            {
                std::uint64_t size = 0;
                if (!ReadVarint(size) || size > bytes.size() - position) return false;
                out.assign(bytes.begin() + static_cast<std::ptrdiff_t>(position),
                    bytes.begin() + static_cast<std::ptrdiff_t>(position + size));
                position += static_cast<std::size_t>(size);
                return true;
            }

            [[nodiscard]] bool Skip(std::uint32_t wire)
            {
                switch (wire)
                {
                case 0: { std::uint64_t ignored = 0; return ReadVarint(ignored); }
                case 1: if (bytes.size() - position < 8) return false; position += 8; return true;
                case 2: { std::vector<std::byte> ignored; return ReadBytes(ignored); }
                case 5: if (bytes.size() - position < 4) return false; position += 4; return true;
                default: return false;
                }
            }
        };

        const auto stringFromBytes = [](const std::vector<std::byte>& value)
        {
            return std::string(reinterpret_cast<const char*>(value.data()), value.size());
        };
        const auto elementTypeFromONNX = [](std::uint64_t value) -> std::optional<ElementType>
        {
            switch (value)
            {
            case 1: return ElementType::Float32;
            case 2: return ElementType::UInt8;
            case 6: return ElementType::Int32;
            case 7: return ElementType::Int64;
            case 9: return ElementType::Bool;
            case 11: return ElementType::Float64;
            default: return std::nullopt;
            }
        };
        const auto opFromName = [](std::string_view name)
        {
            if (name == "Add") return OpKind::Add; if (name == "Sub") return OpKind::Sub;
            if (name == "Mul") return OpKind::Mul; if (name == "Div") return OpKind::Div;
            if (name == "Gemm") return OpKind::Gemm; if (name == "MatMul") return OpKind::MatMul;
            if (name == "Relu") return OpKind::Relu; if (name == "Gelu") return OpKind::Gelu;
            if (name == "Sigmoid") return OpKind::Sigmoid; if (name == "Softmax") return OpKind::Softmax;
            if (name == "Conv") return OpKind::Conv; if (name == "MaxPool") return OpKind::MaxPool;
            if (name == "Flatten") return OpKind::Flatten; if (name == "Reshape") return OpKind::Reshape;
            if (name == "Transpose") return OpKind::Transpose;
            if (name == "LayerNormalization") return OpKind::LayerNormalization;
            if (name == "Gather") return OpKind::Gather; if (name == "Slice") return OpKind::Slice;
            if (name == "Concat") return OpKind::Concat;
            return OpKind::Unknown;
        };

        const auto parseTensor = [&](const std::vector<std::byte>& message, TensorInfo& tensor) -> bool
        {
            Reader reader{ message };
            for (; reader.position < message.size();)
            {
                std::uint32_t field = 0, wire = 0;
                if (!reader.ReadField(field, wire)) return false;
                if (field == 1 && wire == 0) { std::uint64_t dim = 0; if (!reader.ReadVarint(dim) || dim > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) return false; tensor.shape.push_back(static_cast<std::int64_t>(dim)); }
                else if (field == 1 && wire == 2)
                {
                    std::vector<std::byte> packed;
                    if (!reader.ReadBytes(packed)) return false;
                    Reader dimensions{ packed };
                    while (dimensions.position < packed.size())
                    {
                        std::uint64_t dim = 0;
                        if (!dimensions.ReadVarint(dim) || dim > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) return false;
                        tensor.shape.push_back(static_cast<std::int64_t>(dim));
                    }
                }
                else if (field == 2 && wire == 0) { std::uint64_t type = 0; if (!reader.ReadVarint(type)) return false; auto converted = elementTypeFromONNX(type); if (!converted) return false; tensor.elementType = *converted; }
                else if (field == 8 && wire == 2) { std::vector<std::byte> name; if (!reader.ReadBytes(name)) return false; tensor.name = stringFromBytes(name); }
                else if (field == 9 && wire == 2) { if (!reader.ReadBytes(tensor.rawData)) return false; }
                else if (!reader.Skip(wire)) return false;
            }
            return !tensor.name.empty();
        };

        const auto parseAttribute = [&](const std::vector<std::byte>& message, Attribute& attribute) -> bool
        {
            Reader reader{ message };
            std::vector<std::string> values;
            for (; reader.position < message.size();)
            {
                std::uint32_t field = 0, wire = 0;
                if (!reader.ReadField(field, wire)) return false;
                if (field == 1 && wire == 2)
                {
                    std::vector<std::byte> name;
                    if (!reader.ReadBytes(name)) return false;
                    attribute.name = stringFromBytes(name);
                }
                else if (field == 2 && wire == 5)
                {
                    std::uint32_t bits = 0;
                    if (!reader.ReadFixed32(bits)) return false;
                    values.push_back(std::to_string(std::bit_cast<float>(bits)));
                }
                else if (field == 3 && wire == 0)
                {
                    std::uint64_t value = 0;
                    if (!reader.ReadVarint(value)) return false;
                    values.push_back(std::to_string(static_cast<std::int64_t>(value)));
                }
                else if (field == 4 && wire == 2)
                {
                    std::vector<std::byte> value;
                    if (!reader.ReadBytes(value)) return false;
                    values.push_back(stringFromBytes(value));
                }
                else if (field == 7 && (wire == 5 || wire == 2))
                {
                    if (wire == 5)
                    {
                        std::uint32_t bits = 0;
                        if (!reader.ReadFixed32(bits)) return false;
                        values.push_back(std::to_string(std::bit_cast<float>(bits)));
                    }
                    else
                    {
                        std::vector<std::byte> packed;
                        if (!reader.ReadBytes(packed) || packed.size() % 4 != 0) return false;
                        Reader packedReader{ packed };
                        while (packedReader.position < packed.size())
                        {
                            std::uint32_t bits = 0;
                            if (!packedReader.ReadFixed32(bits)) return false;
                            values.push_back(std::to_string(std::bit_cast<float>(bits)));
                        }
                    }
                }
                else if (field == 8 && (wire == 0 || wire == 2))
                {
                    if (wire == 0)
                    {
                        std::uint64_t value = 0;
                        if (!reader.ReadVarint(value)) return false;
                        values.push_back(std::to_string(static_cast<std::int64_t>(value)));
                    }
                    else
                    {
                        std::vector<std::byte> packed;
                        if (!reader.ReadBytes(packed)) return false;
                        Reader packedReader{ packed };
                        while (packedReader.position < packed.size())
                        {
                            std::uint64_t value = 0;
                            if (!packedReader.ReadVarint(value)) return false;
                            values.push_back(std::to_string(static_cast<std::int64_t>(value)));
                        }
                    }
                }
                else if (!reader.Skip(wire)) return false;
            }
            if (attribute.name.empty()) return false;
            for (std::size_t index = 0; index < values.size(); ++index)
            {
                if (index != 0) attribute.value += ',';
                attribute.value += values[index];
            }
            return !values.empty();
        };

        const auto parseNode = [&](const std::vector<std::byte>& message, Node& node) -> bool
        {
            Reader reader{ message };
            std::string opName;
            for (; reader.position < message.size();)
            {
                std::uint32_t field = 0, wire = 0;
                if (!reader.ReadField(field, wire)) return false;
                if ((field == 1 || field == 2 || field == 3 || field == 4) && wire == 2)
                {
                    std::vector<std::byte> value; if (!reader.ReadBytes(value)) return false;
                    const std::string stringValue = stringFromBytes(value);
                    if (field == 1) node.inputs.push_back(stringValue);
                    else if (field == 2) node.outputs.push_back(stringValue);
                    else if (field == 3) node.name = stringValue;
                    else opName = stringValue;
                }
                else if (field == 5 && wire == 2)
                {
                    std::vector<std::byte> nested;
                    Attribute attribute;
                    if (!reader.ReadBytes(nested) || !parseAttribute(nested, attribute)) return false;
                    node.attributes.push_back(std::move(attribute));
                }
                else if (!reader.Skip(wire)) return false;
            }
            node.op = opFromName(opName);
            if (node.name.empty()) node.name = opName;
            return !opName.empty();
        };

        const auto parseDimension = [&](const std::vector<std::byte>& message, std::int64_t& dimension) -> bool
        {
            Reader reader{ message };
            for (; reader.position < message.size();)
            {
                std::uint32_t field = 0, wire = 0;
                if (!reader.ReadField(field, wire)) return false;
                if (field == 1 && wire == 0)
                {
                    std::uint64_t value = 0;
                    if (!reader.ReadVarint(value)
                        || value > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))
                        return false;
                    dimension = static_cast<std::int64_t>(value);
                }
                else if (field == 2 && wire == 2)
                {
                    std::vector<std::byte> symbolic;
                    if (!reader.ReadBytes(symbolic)) return false;
                    dimension = -1;
                }
                else if (!reader.Skip(wire)) return false;
            }
            return dimension != 0;
        };

        const auto parseTensorType = [&](const std::vector<std::byte>& message, TensorInfo& tensor) -> bool
        {
            Reader reader{ message };
            for (; reader.position < message.size();)
            {
                std::uint32_t field = 0, wire = 0;
                if (!reader.ReadField(field, wire)) return false;
                if (field == 1 && wire == 0)
                {
                    std::uint64_t type = 0;
                    if (!reader.ReadVarint(type)) return false;
                    const auto converted = elementTypeFromONNX(type);
                    if (!converted) return false;
                    tensor.elementType = *converted;
                }
                else if (field == 2 && wire == 2)
                {
                    std::vector<std::byte> shapeMessage;
                    if (!reader.ReadBytes(shapeMessage)) return false;
                    Reader shapeReader{ shapeMessage };
                    while (shapeReader.position < shapeMessage.size())
                    {
                        std::uint32_t shapeField = 0, shapeWire = 0;
                        if (!shapeReader.ReadField(shapeField, shapeWire)) return false;
                        if (shapeField == 1 && shapeWire == 2)
                        {
                            std::vector<std::byte> nested;
                            std::int64_t dimension = 0;
                            if (!shapeReader.ReadBytes(nested)
                                || !parseDimension(nested, dimension)) return false;
                            tensor.shape.push_back(dimension);
                        }
                        else if (!shapeReader.Skip(shapeWire)) return false;
                    }
                }
                else if (!reader.Skip(wire)) return false;
            }
            return true;
        };

        const auto parseValueInfo = [&](const std::vector<std::byte>& message, TensorInfo& tensor) -> bool
        {
            Reader reader{ message };
            for (; reader.position < message.size();)
            {
                std::uint32_t field = 0, wire = 0;
                if (!reader.ReadField(field, wire)) return false;
                if (field == 1 && wire == 2) { std::vector<std::byte> name; if (!reader.ReadBytes(name)) return false; tensor.name = stringFromBytes(name); }
                else if (field == 2 && wire == 2)
                {
                    std::vector<std::byte> typeMessage;
                    if (!reader.ReadBytes(typeMessage)) return false;
                    Reader typeReader{ typeMessage };
                    while (typeReader.position < typeMessage.size())
                    {
                        std::uint32_t typeField = 0, typeWire = 0;
                        if (!typeReader.ReadField(typeField, typeWire)) return false;
                        if (typeField == 1 && typeWire == 2)
                        {
                            std::vector<std::byte> tensorType;
                            if (!typeReader.ReadBytes(tensorType)
                                || !parseTensorType(tensorType, tensor)) return false;
                        }
                        else if (!typeReader.Skip(typeWire)) return false;
                    }
                }
                else if (!reader.Skip(wire)) return false;
            }
            return !tensor.name.empty();
        };

        const auto parseGraph = [&](const std::vector<std::byte>& message) -> bool
        {
            Reader reader{ message };
            for (; reader.position < message.size();)
            {
                std::uint32_t field = 0, wire = 0;
                if (!reader.ReadField(field, wire)) return false;
                if ((field == 1 || field == 5 || field == 11 || field == 12 || field == 13)
                    && wire == 2)
                {
                    std::vector<std::byte> nested; if (!reader.ReadBytes(nested)) return false;
                    if (field == 1) { Node node; if (!parseNode(nested, node)) return false; result.graph.nodes.push_back(std::move(node)); }
                    else { TensorInfo tensor; const bool parsed = field == 5 ? parseTensor(nested, tensor) : parseValueInfo(nested, tensor); if (!parsed) return false;
                        if (field == 5) result.graph.initializers.push_back(std::move(tensor));
                        else if (field == 11) result.graph.inputs.push_back(std::move(tensor));
                        else if (field == 12) result.graph.outputs.push_back(std::move(tensor));
                        else result.graph.valueInfo.push_back(std::move(tensor)); }
                }
                else if (!reader.Skip(wire)) return false;
            }
            return true;
        };

        Reader model{ bytes };
        bool foundGraph = false;
        while (model.position < bytes.size())
        {
            std::uint32_t field = 0, wire = 0;
            if (!model.ReadField(field, wire)) { result.diagnostics.push_back("Malformed ONNX protobuf field."); return result; }
            if (field == 7 && wire == 2)
            {
                std::vector<std::byte> graphBytes;
                if (!model.ReadBytes(graphBytes) || !parseGraph(graphBytes)) { result.diagnostics.push_back("Malformed ONNX graph payload."); return result; }
                foundGraph = true;
            }
            else if (field == 8 && wire == 2)
            {
                std::vector<std::byte> message;
                if (!model.ReadBytes(message))
                {
                    result.diagnostics.push_back("Malformed ONNX opset import.");
                    return result;
                }
                Reader opsetReader{ message };
                OperatorSet opset;
                while (opsetReader.position < message.size())
                {
                    std::uint32_t opsetField = 0, opsetWire = 0;
                    if (!opsetReader.ReadField(opsetField, opsetWire))
                    {
                        result.diagnostics.push_back("Malformed ONNX opset field.");
                        return result;
                    }
                    if (opsetField == 1 && opsetWire == 2)
                    {
                        std::vector<std::byte> domain;
                        if (!opsetReader.ReadBytes(domain)) return result;
                        opset.domain = stringFromBytes(domain);
                    }
                    else if (opsetField == 2 && opsetWire == 0)
                    {
                        std::uint64_t version = 0;
                        if (!opsetReader.ReadVarint(version)
                            || version > static_cast<std::uint64_t>(
                                std::numeric_limits<std::int64_t>::max()))
                            return result;
                        opset.version = static_cast<std::int64_t>(version);
                    }
                    else if (!opsetReader.Skip(opsetWire)) return result;
                }
                if (opset.version <= 0)
                {
                    result.diagnostics.push_back("ONNX opset version must be positive.");
                    return result;
                }
                result.graph.operatorSets.push_back(std::move(opset));
            }
            else if (!model.Skip(wire)) { result.diagnostics.push_back("Unsupported ONNX protobuf wire type."); return result; }
        }
        if (!foundGraph) { result.diagnostics.push_back("ONNX model has no graph field."); return result; }
        result.diagnostics = ValidateGraph(result.graph);
        return result;
    }
}
