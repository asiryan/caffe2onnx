import caffe2onnx.src.c2oObject as Node


def getDropoutAttri(layer):

    ratio = layer.dropout_param.dropout_ratio
    ratio = 0.0

    dict = {"ratio":ratio}
    return dict


def getDropoutOutShape(input_shape):
    output_shape = input_shape
    return output_shape

    
def createDropout(layer, nodename, inname, outname, input_shape):
    dict = getDropoutAttri(layer)
    output_shape = getDropoutOutShape(input_shape)
    node = Node.c2oNode(layer, nodename, "Dropout", inname, outname, input_shape, output_shape, dict=dict)
    return node
