module;

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

export module Kairo.ONNX.Runtime;

export import Kairo.ONNX;
import Kairo.Foundation.Math.Tensor;

export namespace kairo::onnx
{
    using RuntimeValue = std::variant<Tensor<float>, Tensor<std::int64_t>>;
    using RuntimeBindings = std::unordered_map<std::string, RuntimeValue>;

    struct ExecutionResult final
    {
        RuntimeBindings outputs;
        RuntimeBindings values;
    };

    [[nodiscard]] inline std::vector<std::int64_t> Int64InitializerValues(
        const TensorInfo& initializer)
    {
        if (initializer.elementType != ElementType::Int64
            || initializer.rawData.size() % sizeof(std::int64_t) != 0)
            throw std::invalid_argument("Expected an Int64 raw_data initializer.");
        std::vector<std::int64_t> values(
            initializer.rawData.size() / sizeof(std::int64_t));
        std::memcpy(values.data(), initializer.rawData.data(), initializer.rawData.size());
        return values;
    }

    namespace runtime_detail
    {
        [[nodiscard]] inline const Attribute* FindAttribute(
            const Node& node, std::string_view name) noexcept
        {
            for (const Attribute& attribute : node.attributes)
                if (attribute.name == name) return &attribute;
            return nullptr;
        }

        [[nodiscard]] inline std::int64_t IntAttribute(
            const Node& node, std::string_view name, std::int64_t fallback)
        {
            const Attribute* attribute = FindAttribute(node, name);
            if (!attribute) return fallback;
            std::size_t parsed = 0;
            const std::int64_t value = std::stoll(attribute->value, &parsed);
            if (parsed != attribute->value.size())
                throw std::invalid_argument("Invalid integer ONNX attribute: " + attribute->name);
            return value;
        }

        [[nodiscard]] inline float FloatAttribute(
            const Node& node, std::string_view name, float fallback)
        {
            const Attribute* attribute = FindAttribute(node, name);
            if (!attribute) return fallback;
            std::size_t parsed = 0;
            const float value = std::stof(attribute->value, &parsed);
            if (parsed != attribute->value.size())
                throw std::invalid_argument("Invalid float ONNX attribute: " + attribute->name);
            return value;
        }

        [[nodiscard]] inline std::vector<std::int64_t> IntListAttribute(
            const Node& node, std::string_view name, std::vector<std::int64_t> fallback)
        {
            const Attribute* attribute = FindAttribute(node, name);
            if (!attribute) return fallback;
            std::vector<std::int64_t> result;
            std::size_t begin = 0;
            while (begin <= attribute->value.size())
            {
                const std::size_t end = attribute->value.find(',', begin);
                const std::string token = attribute->value.substr(
                    begin, end == std::string::npos ? end : end - begin);
                if (token.empty()) throw std::invalid_argument("Empty ONNX list attribute item.");
                result.push_back(std::stoll(token));
                if (end == std::string::npos) break;
                begin = end + 1;
            }
            return result;
        }

        [[nodiscard]] inline std::size_t NormalizeAxis(
            std::int64_t axis, std::size_t rank, bool allowEnd = false)
        {
            const std::int64_t upper = static_cast<std::int64_t>(rank) + (allowEnd ? 1 : 0);
            if (axis < 0) axis += static_cast<std::int64_t>(rank);
            if (axis < 0 || axis >= upper)
                throw std::out_of_range("ONNX axis is outside tensor rank.");
            return static_cast<std::size_t>(axis);
        }

        [[nodiscard]] inline std::vector<std::size_t> BroadcastShape(
            const Tensor<float>& lhs, const Tensor<float>& rhs)
        {
            const std::size_t rank = std::max(lhs.Rank(), rhs.Rank());
            std::vector<std::size_t> shape(rank, 1);
            for (std::size_t trailing = 0; trailing < rank; ++trailing)
            {
                const std::size_t left = trailing < lhs.Rank()
                    ? lhs.Dim(lhs.Rank() - 1 - trailing) : 1;
                const std::size_t right = trailing < rhs.Rank()
                    ? rhs.Dim(rhs.Rank() - 1 - trailing) : 1;
                if (left != right && left != 1 && right != 1)
                    throw std::invalid_argument("ONNX elementwise broadcast dimensions conflict.");
                shape[rank - 1 - trailing] = std::max(left, right);
            }
            return shape;
        }

        template<typename Operation>
        [[nodiscard]] Tensor<float> BroadcastBinary(
            const Tensor<float>& lhs, const Tensor<float>& rhs, Operation operation)
        {
            const std::vector<std::size_t> shape = BroadcastShape(lhs, rhs);
            Tensor<float> output(shape, 0.0f);
            std::vector<std::size_t> coordinates(shape.size(), 0);
            std::vector<std::size_t> left(lhs.Rank(), 0), right(rhs.Rank(), 0);
            for (std::size_t linear = 0; linear < output.Size(); ++linear)
            {
                std::size_t remainder = linear;
                for (std::size_t axis = shape.size(); axis-- > 0;)
                {
                    coordinates[axis] = remainder % shape[axis];
                    remainder /= shape[axis];
                }
                const std::size_t leftOffset = shape.size() - lhs.Rank();
                for (std::size_t axis = 0; axis < lhs.Rank(); ++axis)
                    left[axis] = lhs.Dim(axis) == 1 ? 0 : coordinates[leftOffset + axis];
                const std::size_t rightOffset = shape.size() - rhs.Rank();
                for (std::size_t axis = 0; axis < rhs.Rank(); ++axis)
                    right[axis] = rhs.Dim(axis) == 1 ? 0 : coordinates[rightOffset + axis];
                output[linear] = operation(lhs.At(left), rhs.At(right));
            }
            return output;
        }

        [[nodiscard]] inline Tensor<float> Transpose(
            const Tensor<float>& input, const std::vector<std::int64_t>& requested)
        {
            std::vector<std::size_t> permutation;
            if (requested.empty())
                for (std::size_t axis = input.Rank(); axis-- > 0;) permutation.push_back(axis);
            else
            {
                if (requested.size() != input.Rank())
                    throw std::invalid_argument("Transpose permutation rank mismatch.");
                std::unordered_set<std::size_t> unique;
                for (std::int64_t axis : requested)
                {
                    const std::size_t normalized = NormalizeAxis(axis, input.Rank());
                    if (!unique.emplace(normalized).second)
                        throw std::invalid_argument("Transpose permutation contains duplicates.");
                    permutation.push_back(normalized);
                }
            }
            std::vector<std::size_t> shape(input.Rank());
            for (std::size_t axis = 0; axis < input.Rank(); ++axis)
                shape[axis] = input.Dim(permutation[axis]);
            Tensor<float> output(shape, 0.0f);
            std::vector<std::size_t> outputCoordinates(input.Rank(), 0);
            std::vector<std::size_t> inputCoordinates(input.Rank(), 0);
            for (std::size_t linear = 0; linear < output.Size(); ++linear)
            {
                std::size_t remainder = linear;
                for (std::size_t axis = shape.size(); axis-- > 0;)
                {
                    outputCoordinates[axis] = remainder % shape[axis];
                    remainder /= shape[axis];
                }
                for (std::size_t axis = 0; axis < input.Rank(); ++axis)
                    inputCoordinates[permutation[axis]] = outputCoordinates[axis];
                output[linear] = input.At(inputCoordinates);
            }
            return output;
        }

        [[nodiscard]] inline Tensor<float> ConvNCHW(
            const Tensor<float>& input,
            const Tensor<float>& weights,
            const Node& node)
        {
            if (input.Rank() != 4 || weights.Rank() != 4
                || input.Dim(1) != weights.Dim(1))
                throw std::invalid_argument("ONNX Conv expects NCHW and OIHW tensors.");
            const auto strides = IntListAttribute(node, "strides", { 1, 1 });
            const auto pads = IntListAttribute(node, "pads", { 0, 0, 0, 0 });
            const auto dilations = IntListAttribute(node, "dilations", { 1, 1 });
            if (strides.size() != 2 || pads.size() != 4 || dilations != std::vector<std::int64_t>{1, 1}
                || IntAttribute(node, "group", 1) != 1)
                throw std::invalid_argument("ONNX Conv currently requires 2D group=1 dilation=1.");
            const std::size_t paddedHeight = input.Dim(2) + pads[0] + pads[2];
            const std::size_t paddedWidth = input.Dim(3) + pads[1] + pads[3];
            Tensor<float> padded({ input.Dim(0), input.Dim(1), paddedHeight, paddedWidth }, 0.0f);
            for (std::size_t n = 0; n < input.Dim(0); ++n)
                for (std::size_t c = 0; c < input.Dim(1); ++c)
                    for (std::size_t y = 0; y < input.Dim(2); ++y)
                        for (std::size_t x = 0; x < input.Dim(3); ++x)
                            padded.At({ n, c, y + static_cast<std::size_t>(pads[0]),
                                x + static_cast<std::size_t>(pads[1]) }) = input.At({ n, c, y, x });
            const std::size_t outputHeight =
                (paddedHeight - weights.Dim(2)) / static_cast<std::size_t>(strides[0]) + 1;
            const std::size_t outputWidth =
                (paddedWidth - weights.Dim(3)) / static_cast<std::size_t>(strides[1]) + 1;
            Tensor<float> output({ input.Dim(0), weights.Dim(0), outputHeight, outputWidth }, 0.0f);
            for (std::size_t n = 0; n < output.Dim(0); ++n)
                for (std::size_t o = 0; o < output.Dim(1); ++o)
                    for (std::size_t y = 0; y < outputHeight; ++y)
                        for (std::size_t x = 0; x < outputWidth; ++x)
                            for (std::size_t c = 0; c < input.Dim(1); ++c)
                                for (std::size_t ky = 0; ky < weights.Dim(2); ++ky)
                                    for (std::size_t kx = 0; kx < weights.Dim(3); ++kx)
                                        output.At({ n, o, y, x }) += padded.At({
                                            n, c,
                                            y * static_cast<std::size_t>(strides[0]) + ky,
                                            x * static_cast<std::size_t>(strides[1]) + kx
                                        }) * weights.At({ o, c, ky, kx });
            return output;
        }

        [[nodiscard]] inline Tensor<float> MaxPoolNCHW(
            const Tensor<float>& input, const Node& node)
        {
            if (input.Rank() != 4)
                throw std::invalid_argument("ONNX MaxPool expects NCHW rank-4 input.");
            const auto kernel = IntListAttribute(node, "kernel_shape", {});
            const auto strides = IntListAttribute(node, "strides", { 1, 1 });
            const auto pads = IntListAttribute(node, "pads", { 0, 0, 0, 0 });
            if (kernel.size() != 2 || strides.size() != 2 || pads.size() != 4
                || kernel[0] <= 0 || kernel[1] <= 0 || strides[0] <= 0 || strides[1] <= 0
                || std::any_of(pads.begin(), pads.end(), [](std::int64_t value) { return value < 0; }))
                throw std::invalid_argument("ONNX MaxPool has invalid 2D attributes.");
            const std::size_t paddedHeight = input.Dim(2) + pads[0] + pads[2];
            const std::size_t paddedWidth = input.Dim(3) + pads[1] + pads[3];
            const std::size_t outputHeight =
                (paddedHeight - static_cast<std::size_t>(kernel[0]))
                    / static_cast<std::size_t>(strides[0]) + 1;
            const std::size_t outputWidth =
                (paddedWidth - static_cast<std::size_t>(kernel[1]))
                    / static_cast<std::size_t>(strides[1]) + 1;
            Tensor<float> output({
                input.Dim(0), input.Dim(1), outputHeight, outputWidth
            }, -std::numeric_limits<float>::infinity());
            for (std::size_t n = 0; n < output.Dim(0); ++n)
                for (std::size_t c = 0; c < output.Dim(1); ++c)
                    for (std::size_t oy = 0; oy < outputHeight; ++oy)
                        for (std::size_t ox = 0; ox < outputWidth; ++ox)
                            for (std::size_t ky = 0; ky < static_cast<std::size_t>(kernel[0]); ++ky)
                                for (std::size_t kx = 0; kx < static_cast<std::size_t>(kernel[1]); ++kx)
                                {
                                    const std::int64_t y =
                                        static_cast<std::int64_t>(oy * strides[0] + ky) - pads[0];
                                    const std::int64_t x =
                                        static_cast<std::int64_t>(ox * strides[1] + kx) - pads[1];
                                    if (y >= 0 && x >= 0
                                        && y < static_cast<std::int64_t>(input.Dim(2))
                                        && x < static_cast<std::int64_t>(input.Dim(3)))
                                        output.At({ n, c, oy, ox }) = std::max(
                                            output.At({ n, c, oy, ox }),
                                            input.At({ n, c, static_cast<std::size_t>(y),
                                                static_cast<std::size_t>(x) }));
                                }
            return output;
        }

        [[nodiscard]] inline Tensor<float> Gather(
            const Tensor<float>& data,
            const Tensor<std::int64_t>& indices,
            std::int64_t requestedAxis)
        {
            const std::size_t axis = NormalizeAxis(requestedAxis, data.Rank());
            std::vector<std::size_t> outputShape;
            outputShape.insert(
                outputShape.end(), data.GetShape().begin(),
                data.GetShape().begin() + static_cast<std::ptrdiff_t>(axis));
            outputShape.insert(
                outputShape.end(), indices.GetShape().begin(), indices.GetShape().end());
            outputShape.insert(
                outputShape.end(),
                data.GetShape().begin() + static_cast<std::ptrdiff_t>(axis + 1),
                data.GetShape().end());
            Tensor<float> output(outputShape, 0.0f);
            std::vector<std::size_t> outputCoordinates(output.Rank(), 0);
            std::vector<std::size_t> dataCoordinates(data.Rank(), 0);
            std::vector<std::size_t> indexCoordinates(indices.Rank(), 0);
            for (std::size_t linear = 0; linear < output.Size(); ++linear)
            {
                std::size_t remainder = linear;
                for (std::size_t dimension = output.Rank(); dimension-- > 0;)
                {
                    outputCoordinates[dimension] = remainder % output.Dim(dimension);
                    remainder /= output.Dim(dimension);
                }
                for (std::size_t dimension = 0; dimension < axis; ++dimension)
                    dataCoordinates[dimension] = outputCoordinates[dimension];
                for (std::size_t dimension = 0; dimension < indices.Rank(); ++dimension)
                    indexCoordinates[dimension] = outputCoordinates[axis + dimension];
                std::int64_t selected = indices.At(indexCoordinates);
                if (selected < 0) selected += static_cast<std::int64_t>(data.Dim(axis));
                if (selected < 0 || selected >= static_cast<std::int64_t>(data.Dim(axis)))
                    throw std::out_of_range("ONNX Gather index is out of range.");
                dataCoordinates[axis] = static_cast<std::size_t>(selected);
                for (std::size_t dimension = axis + 1; dimension < data.Rank(); ++dimension)
                    dataCoordinates[dimension] =
                        outputCoordinates[dimension - 1 + indices.Rank()];
                output[linear] = data.At(dataCoordinates);
            }
            return output;
        }

        [[nodiscard]] inline Tensor<float> Concat(
            const std::vector<const Tensor<float>*>& inputs, std::int64_t requestedAxis)
        {
            if (inputs.empty()) throw std::invalid_argument("ONNX Concat requires inputs.");
            const std::size_t axis = NormalizeAxis(requestedAxis, inputs[0]->Rank());
            std::vector<std::size_t> shape = inputs[0]->GetShape();
            shape[axis] = 0;
            for (const Tensor<float>* input : inputs)
            {
                if (!input || input->Rank() != shape.size())
                    throw std::invalid_argument("ONNX Concat rank mismatch.");
                for (std::size_t dimension = 0; dimension < shape.size(); ++dimension)
                    if (dimension != axis && input->Dim(dimension) != inputs[0]->Dim(dimension))
                        throw std::invalid_argument("ONNX Concat non-axis shape mismatch.");
                shape[axis] += input->Dim(axis);
            }
            Tensor<float> output(shape, 0.0f);
            std::vector<std::size_t> coordinates(shape.size(), 0);
            std::size_t axisOffset = 0;
            for (const Tensor<float>* input : inputs)
            {
                std::vector<std::size_t> inputCoordinates(input->Rank(), 0);
                for (std::size_t linear = 0; linear < input->Size(); ++linear)
                {
                    std::size_t remainder = linear;
                    for (std::size_t dimension = input->Rank(); dimension-- > 0;)
                    {
                        inputCoordinates[dimension] = remainder % input->Dim(dimension);
                        remainder /= input->Dim(dimension);
                    }
                    coordinates = inputCoordinates;
                    coordinates[axis] += axisOffset;
                    output.At(coordinates) = (*input)[linear];
                }
                axisOffset += input->Dim(axis);
            }
            return output;
        }

        [[nodiscard]] inline const Tensor<std::int64_t>& Int64Input(
            const RuntimeBindings& values, const Node& node, std::size_t index)
        {
            if (index >= node.inputs.size() || !values.contains(node.inputs[index]))
                throw std::invalid_argument("Missing Int64 ONNX node input.");
            const auto* tensor =
                std::get_if<Tensor<std::int64_t>>(&values.at(node.inputs[index]));
            if (!tensor) throw std::invalid_argument("ONNX node expected Int64 input.");
            return *tensor;
        }
    }

    /// Executes a validated, topologically ordered ONNX graph using native
    /// KairoMath CPU reference kernels. All failures identify the offending
    /// graph contract before returning partial outputs.
    [[nodiscard]] inline ExecutionResult ExecuteGraph(
        const Graph& graph, const RuntimeBindings& feeds)
    {
        using namespace runtime_detail;
        const std::vector<std::string> diagnostics = ValidateGraph(graph);
        if (!diagnostics.empty()) throw std::invalid_argument(diagnostics.front());
        RuntimeBindings values = feeds;
        for (const TensorInfo& input : graph.inputs)
            if (!values.contains(input.name)
                && std::none_of(graph.initializers.begin(), graph.initializers.end(),
                    [&](const TensorInfo& item) { return item.name == input.name; }))
                throw std::invalid_argument("Missing ONNX graph input: " + input.name);
        for (const TensorInfo& initializer : graph.initializers)
        {
            if (values.contains(initializer.name))
                throw std::invalid_argument("ONNX feed shadows initializer: " + initializer.name);
            if (initializer.elementType == ElementType::Float32)
                values.emplace(initializer.name, Float32InitializerTensor(initializer));
            else if (initializer.elementType == ElementType::Int64)
            {
                std::vector<std::size_t> shape;
                for (std::int64_t dimension : initializer.shape)
                    if (dimension <= 0) throw std::invalid_argument("Invalid Int64 initializer shape.");
                    else shape.push_back(static_cast<std::size_t>(dimension));
                values.emplace(initializer.name,
                    Tensor<std::int64_t>(std::move(shape), Int64InitializerValues(initializer)));
            }
            else throw std::invalid_argument("Unsupported runtime initializer dtype.");
        }
        const auto floatInput = [&](const Node& node, std::size_t index) -> const Tensor<float>&
        {
            if (index >= node.inputs.size() || !values.contains(node.inputs[index]))
                throw std::invalid_argument("Missing input for ONNX node: " + node.name);
            const auto* tensor = std::get_if<Tensor<float>>(&values.at(node.inputs[index]));
            if (!tensor) throw std::invalid_argument("ONNX node expected Float32 input: " + node.name);
            return *tensor;
        };
        for (const Node& node : graph.nodes)
        {
            if (node.outputs.size() != 1)
                throw std::invalid_argument("Runtime currently requires one output per ONNX node.");
            if (values.contains(node.outputs[0]))
                throw std::invalid_argument("Duplicate ONNX value producer: " + node.outputs[0]);
            Tensor<float> output;
            switch (node.op)
            {
            case OpKind::Add:
                output = BroadcastBinary(floatInput(node, 0), floatInput(node, 1),
                    [](float a, float b) { return a + b; }); break;
            case OpKind::Sub:
                output = BroadcastBinary(floatInput(node, 0), floatInput(node, 1),
                    [](float a, float b) { return a - b; }); break;
            case OpKind::Mul:
                output = BroadcastBinary(floatInput(node, 0), floatInput(node, 1),
                    [](float a, float b) { return a * b; }); break;
            case OpKind::Div:
                output = BroadcastBinary(floatInput(node, 0), floatInput(node, 1),
                    [](float a, float b)
                    {
                        if (b == 0.0f) throw std::domain_error("ONNX Div by zero.");
                        return a / b;
                    }); break;
            case OpKind::MatMul:
                output = floatInput(node, 0).Rank() == 2
                    ? MatMul(floatInput(node, 0), floatInput(node, 1))
                    : BatchedMatMul(floatInput(node, 0), floatInput(node, 1));
                break;
            case OpKind::Gemm:
            {
                Tensor<float> a = IntAttribute(node, "transA", 0)
                    ? Transpose(floatInput(node, 0), { 1, 0 }) : floatInput(node, 0);
                Tensor<float> b = IntAttribute(node, "transB", 0)
                    ? Transpose(floatInput(node, 1), { 1, 0 }) : floatInput(node, 1);
                output = MatMul(a, b) * FloatAttribute(node, "alpha", 1.0f);
                if (node.inputs.size() > 2)
                    output = BroadcastBinary(output, floatInput(node, 2),
                        [beta = FloatAttribute(node, "beta", 1.0f)](float x, float bias)
                        { return x + beta * bias; });
                break;
            }
            case OpKind::Relu: output = ReLU(floatInput(node, 0)); break;
            case OpKind::Sigmoid: output = Sigmoid(floatInput(node, 0)); break;
            case OpKind::Gelu:
                output = floatInput(node, 0).Map([](float value)
                {
                    return 0.5f * value * (1.0f + std::erf(value / std::sqrt(2.0f)));
                }); break;
            case OpKind::Softmax:
                if (NormalizeAxis(IntAttribute(node, "axis", -1), floatInput(node, 0).Rank())
                    != floatInput(node, 0).Rank() - 1)
                    throw std::invalid_argument("Runtime Softmax currently requires the final axis.");
                output = SoftmaxLastDim(floatInput(node, 0)); break;
            case OpKind::Flatten:
            {
                const Tensor<float>& input = floatInput(node, 0);
                const std::size_t axis = NormalizeAxis(
                    IntAttribute(node, "axis", 1), input.Rank(), true);
                std::size_t outer = 1;
                for (std::size_t index = 0; index < axis; ++index) outer *= input.Dim(index);
                output = input.Contiguous().Reshape({ outer, input.Size() / outer });
                break;
            }
            case OpKind::Reshape:
            {
                const auto* shape = node.inputs.size() > 1 && values.contains(node.inputs[1])
                    ? std::get_if<Tensor<std::int64_t>>(&values.at(node.inputs[1])) : nullptr;
                if (!shape) throw std::invalid_argument("Reshape requires an Int64 shape input.");
                std::vector<std::size_t> requested(shape->Size());
                std::size_t inferred = shape->Size();
                std::size_t known = 1;
                for (std::size_t index = 0; index < shape->Size(); ++index)
                {
                    const std::int64_t dimension = (*shape)[index];
                    if (dimension == -1)
                    {
                        if (inferred != shape->Size())
                            throw std::invalid_argument("Reshape permits one inferred dimension.");
                        inferred = index; requested[index] = 1;
                    }
                    else if (dimension <= 0)
                        throw std::invalid_argument("Reshape dimensions must be positive or -1.");
                    else { requested[index] = static_cast<std::size_t>(dimension); known *= requested[index]; }
                }
                if (inferred != shape->Size())
                {
                    if (known == 0 || floatInput(node, 0).Size() % known != 0)
                        throw std::invalid_argument("Reshape inferred dimension is incompatible.");
                    requested[inferred] = floatInput(node, 0).Size() / known;
                }
                output = floatInput(node, 0).Contiguous().Reshape(std::move(requested));
                break;
            }
            case OpKind::Transpose:
                output = Transpose(floatInput(node, 0), IntListAttribute(node, "perm", {})); break;
            case OpKind::LayerNormalization:
            {
                output = LayerNormLastDim(
                    floatInput(node, 0), FloatAttribute(node, "epsilon", 1e-5f));
                if (node.inputs.size() > 1)
                    output = BroadcastBinary(output, floatInput(node, 1),
                        [](float a, float b) { return a * b; });
                if (node.inputs.size() > 2)
                    output = BroadcastBinary(output, floatInput(node, 2),
                        [](float a, float b) { return a + b; });
                break;
            }
            case OpKind::Conv:
                output = ConvNCHW(floatInput(node, 0), floatInput(node, 1), node);
                if (node.inputs.size() > 2)
                {
                    const Tensor<float>& bias = floatInput(node, 2);
                    if (bias.Rank() != 1 || bias.Dim(0) != output.Dim(1))
                        throw std::invalid_argument("Conv bias shape mismatch.");
                    for (std::size_t n = 0; n < output.Dim(0); ++n)
                        for (std::size_t c = 0; c < output.Dim(1); ++c)
                            for (std::size_t y = 0; y < output.Dim(2); ++y)
                                for (std::size_t x = 0; x < output.Dim(3); ++x)
                                    output.At({ n, c, y, x }) += bias[c];
                }
                break;
            case OpKind::MaxPool:
                output = MaxPoolNCHW(floatInput(node, 0), node);
                break;
            case OpKind::Gather:
                output = Gather(
                    floatInput(node, 0), Int64Input(values, node, 1),
                    IntAttribute(node, "axis", 0));
                break;
            case OpKind::Concat:
            {
                std::vector<const Tensor<float>*> inputs;
                inputs.reserve(node.inputs.size());
                for (std::size_t index = 0; index < node.inputs.size(); ++index)
                    inputs.push_back(&floatInput(node, index));
                output = Concat(inputs, IntAttribute(node, "axis", 0));
                break;
            }
            case OpKind::Slice:
            {
                const Tensor<float>& data = floatInput(node, 0);
                const Tensor<std::int64_t>& starts = Int64Input(values, node, 1);
                const Tensor<std::int64_t>& ends = Int64Input(values, node, 2);
                if (starts.Size() != ends.Size())
                    throw std::invalid_argument("ONNX Slice starts/ends length mismatch.");
                std::vector<std::int64_t> axes(starts.Size());
                std::vector<std::int64_t> steps(starts.Size(), 1);
                if (node.inputs.size() > 3)
                {
                    const auto& source = Int64Input(values, node, 3);
                    if (source.Size() != starts.Size())
                        throw std::invalid_argument("ONNX Slice axes length mismatch.");
                    for (std::size_t index = 0; index < source.Size(); ++index)
                        axes[index] = source[index];
                }
                else
                    for (std::size_t index = 0; index < axes.size(); ++index)
                        axes[index] = static_cast<std::int64_t>(index);
                if (node.inputs.size() > 4)
                {
                    const auto& source = Int64Input(values, node, 4);
                    if (source.Size() != starts.Size())
                        throw std::invalid_argument("ONNX Slice steps length mismatch.");
                    for (std::size_t index = 0; index < source.Size(); ++index)
                        steps[index] = source[index];
                }
                std::vector<std::int64_t> begin(data.Rank(), 0);
                std::vector<std::int64_t> step(data.Rank(), 1);
                std::vector<std::size_t> shape = data.GetShape();
                for (std::size_t index = 0; index < starts.Size(); ++index)
                {
                    const std::size_t axis = NormalizeAxis(axes[index], data.Rank());
                    if (steps[index] <= 0)
                        throw std::invalid_argument("ONNX Slice currently requires positive steps.");
                    const std::int64_t dimension = static_cast<std::int64_t>(data.Dim(axis));
                    std::int64_t first = starts[index] < 0 ? starts[index] + dimension : starts[index];
                    std::int64_t last = ends[index] < 0 ? ends[index] + dimension : ends[index];
                    first = std::clamp(first, std::int64_t{0}, dimension);
                    last = std::clamp(last, std::int64_t{0}, dimension);
                    begin[axis] = first;
                    step[axis] = steps[index];
                    shape[axis] = last <= first ? 0 : static_cast<std::size_t>(
                        (last - first + steps[index] - 1) / steps[index]);
                }
                output = Tensor<float>(shape, 0.0f);
                std::vector<std::size_t> outputCoordinates(output.Rank(), 0);
                std::vector<std::size_t> inputCoordinates(data.Rank(), 0);
                for (std::size_t linear = 0; linear < output.Size(); ++linear)
                {
                    std::size_t remainder = linear;
                    for (std::size_t axis = output.Rank(); axis-- > 0;)
                    {
                        outputCoordinates[axis] = remainder % output.Dim(axis);
                        remainder /= output.Dim(axis);
                        inputCoordinates[axis] = static_cast<std::size_t>(
                            begin[axis] + step[axis]
                                * static_cast<std::int64_t>(outputCoordinates[axis]));
                    }
                    output[linear] = data.At(inputCoordinates);
                }
                break;
            }
            default:
                throw std::invalid_argument(
                    "ONNX runtime lowering is not implemented for " + OpKindName(node.op));
            }
            values.emplace(node.outputs[0], std::move(output));
        }
        RuntimeBindings outputs;
        for (const TensorInfo& output : graph.outputs)
        {
            const auto iterator = values.find(output.name);
            if (iterator == values.end())
                throw std::invalid_argument("ONNX graph output was not produced: " + output.name);
            outputs.emplace(output.name, iterator->second);
        }
        return { std::move(outputs), std::move(values) };
    }
}
