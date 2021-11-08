from google.protobuf import text_format
import onnx
from caffe2onnx.proto import caffe_upsample_pb2

def loadcaffemodel(net_path, model_path):
    # read prototxt
    net = caffe_upsample_pb2.NetParameter()
    text_format.Merge(open(net_path).read(), net)
    # read caffemodel
    model = caffe_upsample_pb2.NetParameter()
    f = open(model_path, 'rb')
    model.ParseFromString(f.read())
    f.close()
    print("caffe model has been successfully loaded")
    return net, model

def loadonnxmodel(onnx_path):
    onnxmodel = onnx.load(onnx_path)
    return onnxmodel

def saveonnxmodel(onnx_model, onnx_save_path):
    try:
        #onnx.checker.check_model(onnx_model)
        onnx.save_model(onnx_model, onnx_save_path)
        print("onnx model has been successfully saved to " + onnx_save_path)
    except Exception as e:
        print("onnx model was not saved successfully:\n", e)
