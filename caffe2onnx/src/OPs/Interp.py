import caffe2onnx.src.c2oObject as Node
import numpy as np

def get_interp_attri(layer, input_shape):
    height = layer.interp_param.height
    width = layer.interp_param.width
    zoom_factor = layer.interp_param.zoom_factor
    shrink_factor = layer.interp_param.shrink_factor
    pad_beg = layer.interp_param.pad_beg
    pad_end = layer.interp_param.pad_end
    H, W = input_shape[0][2], input_shape[0][3]

    sacles = [1.0, 1.0, 1.0, 1.0]
    if height > H and width > W:
        if height / H == width / W:
            scale = float(height / H)
            scales = [1.0, 1.0, scale, scale]
            attributes = {"mode": "linear",
                          'scales': scales}
            return attributes
    if height == 0 and width == 0:
        if zoom_factor > 1 and shrink_factor == 1:
            height_in_eff = height + pad_beg + pad_end
            width_in_eff = width + pad_beg + pad_end
            height_out = height_in_eff + (height_in_eff - 1) * (zoom_factor -1)
            width_out = width_in_eff + (width_in_eff - 1) * (zoom_factor -1)
            scale_height = float(height_out /height_in_eff)
            scale_width = float(width_out /width_in_eff)
            scales = [1.0, 1.0, scale_height, scale_width]
            attributes = {"mode": "linear",
                          'scales': scales}
            return attributes
        else:
            print("do not support interp type")
            exit(-1)


def get_interp_output_shape(layer, input_shape, attributes):
    scales = attributes.get("scales")
    output_shape = [np.multiply(np.array(scales, dtype=np.int), np.array(input_shape[0])).tolist()]
    return output_shape

def create_interp_node(layer, node_name, input_name, output_name, input_shape):
    attributes = get_interp_attri(layer, input_shape)
    output_shape = get_interp_output_shape(layer, input_shape, attributes)
    node = Node.c2oNode(layer, node_name, "Upsample", input_name, output_name, input_shape, output_shape, attributes)
    return node
