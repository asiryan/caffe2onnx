import caffe2onnx.src.c2oObject as Node


def getBNAttri(layer):
    eps = layer.batch_norm_param.eps
    momentum = layer.batch_norm_param.moving_average_fraction
    dict = {"epsilon": eps,
            "momentum": momentum
            }
    return dict


def getBNOutShape(input_shape):
    output_shape = input_shape
    return output_shape


def createBN(layer, nodename, inname, outname, input_shape):
    dict = getBNAttri(layer)
    output_shape = getBNOutShape(input_shape)
    node = Node.c2oNode(layer, nodename, "BatchNormalization", inname, outname, input_shape, output_shape,dict)
    return node