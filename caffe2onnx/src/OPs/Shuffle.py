import caffe2onnx.src.c2oObject as Node


def getReshapeOutShape(layer, input_shape):
    try:
        re_shape = layer.reshape_param.shape.dim
    except Exception as e:
        re_shape = []

    in_prod = 1
    for dim in input_shape[0]:
        in_prod = in_prod * dim
    if re_shape == []:
        output_shape = [[1, in_prod]]
    else:
        output_shape = re_shape
        for i in range(len(re_shape)):
            if re_shape[i] == 0:
                output_shape[i] = input_shape[0][i]

        for j in range(len(output_shape)):
            if output_shape[j] == -1:
                for d in output_shape:
                    in_prod = in_prod / d
                output_shape[j] = int(in_prod * -1)
        output_shape = [output_shape]
    return output_shape


def createShuffle(layer, nodename, inname, outname, input_shape):
    output_shape = getReshapeOutShape(layer, input_shape)
    node = Node.c2oNode(layer, nodename, "Reshape", inname, outname, input_shape, output_shape)
    return node
