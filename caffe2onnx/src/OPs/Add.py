import caffe2onnx.src.c2oObject as Node


def get_add_output_shape(input_shape):
    output_shape = input_shape[0]
    return [output_shape]


def create_add_node(layer, node_name, input_name, output_name, input_shape):
    output_shape = get_add_output_shape(input_shape)

    node = Node.c2oNode(layer, node_name, 'Add', input_name, output_name, input_shape, output_shape)

    return node
