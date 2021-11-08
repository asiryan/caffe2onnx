import numpy as np
import caffe2onnx.src.c2oObject as Node


def getConvTransposeAttri(layer):

    dilations = [1, 1]
    if layer.convolution_param.dilation != []:
        dilation = layer.convolution_param.dilation[0]
        dilations = [dilation, dilation]

    pads = [0, 0, 0, 0]
    if layer.convolution_param.pad != []:
        pads = np.array([layer.convolution_param.pad] * 4).reshape(1, -1)[0].tolist()
    elif layer.convolution_param.pad_h != 0 or layer.convolution_param.pad_w != 0:
        pads = [layer.convolution_param.pad_h, layer.convolution_param.pad_w, layer.convolution_param.pad_h,
                layer.convolution_param.pad_w]

    strides = [1, 1]
    if layer.convolution_param.stride != []:
        strides = np.array([layer.convolution_param.stride] * 2).reshape(1, -1)[0].tolist()
    elif layer.convolution_param.stride_h != 0 and layer.convolution_param.stride_w != 0:
        strides = [layer.convolution_param.stride_h, layer.convolution_param.stride_w]

    kernel_shape = np.array([layer.convolution_param.kernel_size] * 2).reshape(1, -1)[0].tolist()
    if layer.convolution_param.kernel_size == []:
        kernel_shape = [layer.convolution_param.kernel_h, layer.convolution_param.kernel_w]

    group = layer.convolution_param.group

    dict = {  # "auto_pad":"NOTSET",
        "dilations": dilations,
        "group": group,
        "kernel_shape": kernel_shape,
        "pads": pads,
        "strides": strides
    }
    return dict


def getConvTransposeOutShape(input_shape, layer,dict):
    dilations = dict["dilations"]
    kernel_shape = dict["kernel_shape"]
    pads = dict["pads"]
    strides = dict["strides"]
    kernel_num = layer.convolution_param.num_output

    def get_output_shape(i, k, p, s):
        return (i-1)*s + k -2*p

    h = get_output_shape(input_shape[0][2], kernel_shape[0], pads[0], strides[0])
    w = get_output_shape(input_shape[0][3], kernel_shape[1], pads[1], strides[1])

    output_shape = [[input_shape[0][0], kernel_num, h, w]]

    return output_shape


def createConvTranspose(layer, nodename, inname, outname, input_shape):
    dict = getConvTransposeAttri(layer)
    output_shape = getConvTransposeOutShape(input_shape, layer, dict)
    node = Node.c2oNode(layer, nodename, "ConvTranspose", inname, outname, input_shape, output_shape, dict)
    return node
