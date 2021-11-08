import caffe2onnx.src.c2oObject as Node


def getOutShape(input_shape):
    output_shape = input_shape
    return output_shape


def createSigmoid(layer, nodename, inname, outname, input_shape):
    output_shape = getOutShape(input_shape)

    node = Node.c2oNode(layer, nodename, "Sigmoid", inname, outname, input_shape, output_shape)

    return node
