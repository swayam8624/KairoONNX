import Kairo.ONNX;

#include <cassert>

int main()
{
    assert(kairo::onnx::IsSupportedInferenceOp(kairo::onnx::OpKind::MatMul));
    assert(!kairo::onnx::IsSupportedInferenceOp(kairo::onnx::OpKind::Unknown));
    return 0;
}
