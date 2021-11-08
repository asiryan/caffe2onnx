import caffe2onnx.src.c2oObject as Node
import typing


def getTransposeAttri(layer) -> typing.Dict:
    if layer.type == "ShuffleChannel":

        perm_array = [0, 2, 1, 3, 4]
        attributes = {"perm": perm_array}
        return attributes
    else:
        orders = layer.permute_param.order
        attributes = {"perm": orders}
        return attributes


def getTransposeOutShape(layer, input_shape, attributes):
    if layer.type == "ShuffleChannel":
        n, g, c, h, w = input_shape[0][0], input_shape[0][1], input_shape[0][2], input_shape[0][3], input_shape[0][4]

        output_shape = [[n, c, g, h, w]]
        return output_shape
    else:
        orders = attributes.get("perm")
        shape = []
        for order in orders:
            shape.append(input_shape[0][order])
        return [shape]


def createTranspose(layer, node_name, input_name, output_name, input_shape) -> Node:
    attributes = getTransposeAttri(layer)

    output_shape = getTransposeOutShape(layer, input_shape, attributes)

    node = Node.c2oNode(layer, node_name, "Transpose", input_name, output_name, input_shape, output_shape, attributes)
    return node
