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
