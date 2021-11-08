import caffe2onnx.src.c2oObject as Node
import numpy as np

def get_crop_param(layer, input_shape):
    axis: int = layer.crop_param.axis
    crop_offset = layer.crop_param.offset

    if not crop_offset:
        offset_0 = 0
    else:
        offset_0 = crop_offset[0]

    offset = []
    starts = []
    axes = []
    ends = []

    for i in range(len(input_shape[0])):
        if i < axis:
            start = 0
            end = input_shape[1][i]
        else:
            if (i - axis) >= len(crop_offset):
                offset.append(offset_0)
            else:
                offset.append(crop_offset[i - axis])

            start = offset[i - axis]
            end = start + input_shape[1][i]

        if input_shape[0][i] != input_shape[1][i]:
            axes.append(i)
            starts.append(start)
            ends.append(end)

    return starts, ends, axes


def get_crop_output_shape(layer, input_shape):  
    return [input_shape[1]]


def create_crop_node(layer, node_name, input_name, output_name, input_shape):
    output_shape = get_crop_output_shape(layer, input_shape)
    node = Node.c2oNode(layer, node_name, "Slice", input_name, output_name, input_shape, output_shape)
    return node
