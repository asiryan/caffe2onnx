import caffe2onnx.src.c2oObject as Node


def getSoftmaxAttri(layer):

    axis = layer.softmax_param.axis

    dict = {"axis": axis}
    return dict
    

def getSoftmaxOutShape(input_shape):
    output_shape = input_shape
    return output_shape


def createSoftmax(layer, nodename, inname, outname, input_shape):
    dict = getSoftmaxAttri(layer)
    output_shape = getSoftmaxOutShape(input_shape)

    node = Node.c2oNode(layer, nodename, "Softmax", inname, outname, input_shape, output_shape, dict)
    return node
