# Supported caffe operators

| operator              | caffe2onnx v1.* (onnx 1.4.0)       | caffe2onnx v2.* (onnx 1.6.0)                          |
|-----------------------|:----------------------------------:|:------------------------------------------------:|
| Add                   | -                                  | ✓                                                |
| Axpy                  | -                                  | ✓                                                |
| BatchNorm             | ✓                                  | ✓                                                |
| Clip                  | -                                  | ✓                                                |
| Concat                | ✓                                  | ✓                                                |
| Conv                  | ✓                                  | ✓                                               |
| ConvTranspose         | ✓                                  | ✓                                               |
| Crop                  | -                                  | ✓                                               |
| DetectionOutput       | -                                  | ✓                       |
| Dropout               | ✓                                  | ✓                                                |
| Eltwise               | ✓                                  | ✓                                               |
| Flatten               | -                                   | ✓                                          |
| Gemm                  | ✓                                  | ✓                                                |
| InstanceNorm          | -                                  | ✓                                               |
| Interp                | -                                  | ✓                                                |
| Log                   | -                                  | ✓                                               |
| LpNormalization       | -                                  | ✓                                               |
| LRN                   | ✓                                  | ✓                                                |
| Min                   | -                                  | ✓                                                |
| Mul                   | -                                  | ✓                                                |
| Pooling               | ✓                                  | ✓                                                |
| Power                 | -                                  | ✓                                                |
| PRelu                 | ✓                                  | ✓                                                |
| PriorBox              | -                                  | ✓                                                |
| ReLU                  | ✓                                  | ✓                                                |
| Reshape               | ✓                                  | ✓                                                |
| Resize                | -                                  | ✓                                                |
| Shuffle               | -                                  | ✓                                                |
| Sigmoid               | -                                  | ✓                                                |
| Slice                 | -                                  | ✓                                                |
| Softmax               | ✓                                  | ✓                                                |
| Tanh                  | -                                  | ✓                                                |
| Transpose             | -                                  | ✓                                                |
| UnPooling             | ✓                                  | ✓                                                |
| Upsample              | ✓                                  | ✓                                                |

If you want add an op which is not supported by our repo, you can look up to docs ofr [onnx v1.4.0](https://github.com/onnx/onnx/blob/rel-1.4.0/docs/Operators.md) or [onnx v1.6.0](https://github.com/onnx/onnx/blob/rel-1.6.0/docs/Operators.md) and add your own op by following the other existing ops in src/OPs.
