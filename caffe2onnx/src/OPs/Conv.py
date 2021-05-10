import numpy as np
import src.c2oObject as Node
##---------------------------------------------------Conv-------------------------------------------------------##
# Get hyperparameters
def getConvAttri(layer):
    # Expansion coefficient dilations
    dilations = [1, 1]
    if layer.convolution_param.dilation != []:
        dilation = layer.convolution_param.dilation[0]
        dilations = [dilation, dilation]
    # Fill pads
    pads = [0, 0, 0, 0]  # default is 0
    if layer.convolution_param.pad != []:  # If there is a pad, the value is assigned according to the pad
        pads = np.array([layer.convolution_param.pad] * 4).flatten().tolist()
    elif layer.convolution_param.pad_h != 0 or layer.convolution_param.pad_w != 0:  # If there is pad_w, pad_h is assigned according to it
        pads = [layer.convolution_param.pad_h, layer.convolution_param.pad_w, layer.convolution_param.pad_h,
                layer.convolution_param.pad_w]
    # Strides
    strides = [1, 1]  # default is 1
    if layer.convolution_param.stride != []:
        strides = np.array([layer.convolution_param.stride] * 2).flatten().tolist()
    # Convolution kernel size kernel_shape
    kernel_shape = np.array([layer.convolution_param.kernel_size] * 2).flatten().tolist()
    if layer.convolution_param.kernel_size == []:
        kernel_shape = [layer.convolution_param.kernel_h, layer.convolution_param.kernel_w]
    # Group
    group = layer.convolution_param.group


    # Hyperparameter dictionary
    dict = {  # "auto_pad":"NOTSET",
        "dilations": dilations,
        "group": group,
        "kernel_shape": kernel_shape,
        "pads": pads,
        "strides": strides
    }
    return dict

# Calculate the output dimension
def getConvOutShape(input_shape,layer,dict):
    dilations = dict["dilations"]
    kernel_shape = dict["kernel_shape"]
    pads = dict["pads"]
    strides = dict["strides"]
    ## Number of convolution kernelskernel_num
    kernel_num = layer.convolution_param.num_output

    # Calculate input dimensions output_shape
    h = (input_shape[0][2] - kernel_shape[0] + pads[0] + pads[2] - (kernel_shape[0]-1)*(dilations[0]-1))/strides[0] + 1 
    # Output dimension N = ((input dimension I-convolution kernel dimension K + 2 * padding P-(convolution kernel dimension -1) * (expansion coefficient -1)) / step size S) + 1
    # When h is not an integer and pad is not set, when the output is a non-integer, round up, that is, fill 1 on the right and bottom
    if h > int(h) and layer.convolution_param.pad == []:
        output_shape_h = int(h) + 1
        pads = [0,0,1,1]
    else:
        output_shape_h = int(h)
    
    w = (input_shape[0][3] - kernel_shape[1] + pads[1] + pads[3] - (kernel_shape[1]-1)*(dilations[1]-1))/strides[1] + 1 
    # Output dimension N = ((input dimension I-convolution kernel dimension K + 2 * padding P-(convolution kernel dimension -1) * (expansion coefficient -1)) / step size S) + 1
    # When h is not an integer and pad is not set, when the output is a non-integer, round up, that is, fill 1 on the right and bottom
    if w > int(w) and layer.convolution_param.pad == []:
        output_shape_w = int(w) + 1
        pads = [0,0,1,1]
    else:
        output_shape_w = int(w)

    output_shape = [[input_shape[0][0],kernel_num,output_shape_h,output_shape_w]]

    return output_shape

# Build node
def createConv(layer, nodename, inname, outname, input_shape):
    dict = getConvAttri(layer)
    output_shape = getConvOutShape(input_shape, layer, dict)
    # Build node
    node = Node.c2oNode(layer, nodename, "Conv", inname, outname, input_shape, output_shape, dict)
    print(nodename, " node construction completed")
    return node

