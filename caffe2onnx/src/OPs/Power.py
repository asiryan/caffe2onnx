import caffe2onnx.src.c2oObject as Node
import numpy as np

def get_power_param(layer):
    power: int = layer.power_param.power
    scale: int = layer.power_param.scale
    shift: int = layer.power_param.shift
    return np.array([power]), np.array([scale]), np.array([shift])


def get_power_output_shape(input_shape):
    return [input_shape[0]]


def create_power_node(layer, node_name, input_name, output_name, input_shape):
    output_shape = get_power_output_shape(input_shape)
    node = Node.c2oNode(layer, node_name, "Pow", input_name, output_name, input_shape, output_shape)
    return node
