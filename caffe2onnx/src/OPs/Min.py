import caffe2onnx.src.c2oObject as Node


def get_min_output_shape(input_shape):
    output_shape = input_shape
    return output_shape


def create_min_op(layer, node_name, input_name, output_name, input_shape):
    output_shape = get_min_output_shape(input_shape)
    node = Node.c2oNode(layer, node_name, "Min", input_name, output_name, input_shape, output_shape)
    return node
