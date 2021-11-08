import caffe2onnx.src.c2oObject as Node
from typing import Dict


def create_attribute(layer):
    attribute: Dict = {
        'axis': 1,
        'p': 2
    }
    return attribute


def get_node_output(input_shape):
    output_shape = input_shape
    return output_shape


def create_Lp_Normalization(layer, node_name, input_name, output_name, input_shape):
    attribute = create_attribute(layer)
    output_shape = get_node_output(input_shape)

    node = Node.c2oNode(layer, node_name, "LpNormalization", input_name, output_name, input_shape, output_shape,
                        attribute)
    return node
