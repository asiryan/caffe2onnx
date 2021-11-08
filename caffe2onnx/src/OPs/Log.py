import caffe2onnx.src.c2oObject as Node


def get_log_output_shape(input_shape):
    output_shape = input_shape
    return output_shape


def create_log_node(layer, node_name, input_name, output_name, input_shape):
    output_shape = get_log_output_shape(layer)

    node = Node.c2oNode(layer, node_name, 'Log', input_name, output_name, input_shape, output_shape)

    return node
