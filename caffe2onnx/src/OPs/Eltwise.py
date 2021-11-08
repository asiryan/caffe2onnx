import caffe2onnx.src.c2oObject as Node


def createEltwise(layer, nodename, inname, outname, input_shape):

    if layer.eltwise_param.operation == 0:
        node = __createMul(layer, nodename, inname, outname, input_shape)

    elif layer.eltwise_param.operation == 1:
        node = __createAdd(layer, nodename, inname, outname, input_shape)

    elif layer.eltwise_param.operation == 2:
        node = __createMax(layer, nodename, inname, outname, input_shape)

    return node


def __createMul(layer, nodename, inname, outname, input_shape):
    output_shape = input_shape[0]
    node = Node.c2oNode(layer, nodename, "Mul", inname, outname, input_shape, output_shape)
    return node


def __createAdd(layer, nodename, inname, outname, input_shape):
    output_shape = [input_shape[0]]
    node = Node.c2oNode(layer, nodename, "Add", inname, outname, input_shape, output_shape)
    return node


def __createMax(layer, nodename, inname, outname, input_shape):
    output_shape = input_shape
    node = Node.c2oNode(layer, nodename, "Max", inname, outname, input_shape, output_shape)
    return node
