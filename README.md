# caffe2onnx
This tool converts Caffe models to ONNX via command line (without Caffe environment). 

## Installation
Install from pypi  
```
pip install caffe2onnx
```

Install latest from github  
```
pip install git+https://github.com/asiryan/caffe2onnx
```

Build and install latest from source  
```
git clone https://github.com/asiryan/caffe2onnx
python setup.py install
``` 

## Usage
To get started with **caffe2onnx**, run the *caffe2onnx.convert* command, providing:
* the path to your caffe prototxt,
* the path to your caffe model (*not required*),
* the output path of the onnx model (*not required*),
* frozen graph or not (*not required*).

```
python -m caffe2onnx.convert
    --prototxt          caffe prototxt file path
    [--caffemodel       caffe caffemodel file path]
    [--onnx             output onnx file path]
    [--frozen           frozen graph or not]
```

## Operators
See the documentation of [caffe supported operators](https://github.com/FaceONNX/caffe2onnx/blob/main/docs/caffe-operators.md).

## References
caffe-onnx by [htshinichi](https://github.com/htshinichi/caffe-onnx)  
TNN by [Tencent](https://github.com/Tencent/TNN)

## License
[BSD-3](https://github.com/asiryan/caffe2onnx/blob/main/LICENSE)
