# caffe-onnx
This tool converts Caffe models to ONNX via command line (without Caffe environment).

## Introduction
**caffe-onnx** is based on previous work by [htshinichi](https://github.com/htshinichi/caffe-onnx).  
  
This is the second version of converting caffe model to onnx model. In this version, all the parameters will be transformed to tensor and tensor value info when reading **.caffemodel** file and each operator node is constructed directly into the type of NodeProto in **onnx**.

## Installation
Install from pypi  
```
pip install caffe-onnx
```

Install latest from github  
```
pip install git+https://github.com/asiryan/caffe-onnx
```

Build and install latest from source  
```
git clone https://github.com/asiryan/caffe-onnx
python setup.py install
``` 

## Usage
To get started with **caffe-onnx**, run the *caffe2onnx.convert* command, providing:
* the path to your caffe prototxt,
* the path to your caffe model (*not required*),
* the output path of the onnx model (*not required*),
* frozen graph or not (*not required*).

```
python -m caffe2onnx.convert
    --prototxt          caffe's prototxt file path
    [--caffemodel       caffe's caffemodel file path]
    [--onnx             output onnx file path]
    [--frozen           frozen graph or not]
```

## Operators
* BatchNorm
* Convolution
* Deconvolution
* Concat
* Dropout
* InnerProduct(Reshape+Gemm)
* LRN
* Pooling
* Unpooling
* ReLU
* Softmax
* Eltwise
* Upsample
* Scale

## Models
* ResNet
* AlexNet
* AgeNet
* Yolo V3
* VGG
* GoogleNet
* CaffeNet
* Sphereface

## License
[BSD-3](LICENSE)
