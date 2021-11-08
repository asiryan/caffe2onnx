import caffe2onnx.src.c2oObject as Node


def get_attribute(layer):
    attributes = {}
    max_attribute = 0
    min_attribute = 0
    if layer.type == 'ReLU6':
        max_attribute = 6.0
        min_attribute = 0

    attribute = {
        'max': max_attribute,
        'min': min_attribute
    }
    return attributes


def get_clip_output_shape(input_shape):
    output_shape = input_shape
    return output_shape


def create_clip_node(layer, node_name, input_name, output_name, input_shape):
    output_shape = get_clip_output_shape(input_shape)
    node = Node.c2oNode(layer, node_name, 'Clip', input_name, output_name, input_shape, output_shape)
    return node
