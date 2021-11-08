import caffe2onnx.src.c2oObject as Node


def getLRNAttri(layer):

    size = layer.lrn_param.local_size
    alpha = layer.lrn_param.alpha
    beta = layer.lrn_param.beta

    dict = {"alpha":alpha,
            "beta":beta,
            "bias":1.0,
            "size": size}
    return dict


def getLRNOutShape(input_shape):
    output_shape = input_shape
    return output_shape


def createLRN(layer,nodename, inname,outname,input_shape):
    dict = getLRNAttri(layer)
    output_shape = getLRNOutShape(input_shape)
    node = Node.c2oNode(layer, nodename, "LRN", inname, outname, input_shape, output_shape, dict)
    return node
