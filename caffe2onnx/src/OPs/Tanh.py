import caffe2onnx.src.c2oObject as Node


def getOutShape(input_shape):
    return input_shape


def createTanh(layer, nodename, inname, outname, input_shape):
    output_shape = getOutShape(input_shape)
    node = Node.c2oNode(layer, nodename, "Tanh", inname,
                        outname, input_shape, output_shape)
    return node
