import numpy as np
import caffe2onnx.src.c2oObject as Node
##-----------------------------------------------------Pooling--------------------------------------------------##
# Get hyperparameters
def getPoolingAttri(layer):
    # Pooled core size
    kernel_shape = np.array([layer.pooling_param.kernel_size]*2).flatten().tolist()
    if layer.pooling_param.kernel_size == []:
        kernel_shape = [layer.pooling_param.kernel_h,layer.pooling_param.kernel_w]
    # Step
    strides = [1, 1] # the default is 1
    if layer.pooling_param.stride != []:
        strides = np.array([layer.pooling_param.stride]*2).flatten().tolist()
    # Filling
    pads = [0, 0, 0, 0] # the default is 0
    # Here is the same as when convolution, if there is a pad, it is set according to its value
    if layer.pooling_param.pad != []:
        pads = np.array([layer.pooling_param.pad] * 4).flatten().tolist()
    elif layer.pooling_param.pad_h != 0 or layer.pooling_param.pad_w != 0:
        pads = [layer.pooling_param.pad_h,layer.pooling_param.pad_w,layer.pooling_param.pad_h,layer.pooling_param.pad_w]

    # Hyperparameter dictionary
    dict = {"kernel_shape":kernel_shape,
            "strides":strides,
            "pads":pads
            }
    return dict

# Calculate the output dimension
def getPoolingOutShape(input_shape,layer,dict, with_indices=False):
    kernel_shape = dict["kernel_shape"]
    pads = dict["pads"]
    strides = dict["strides"]

    # Calculate the output dimension, as with convolution, round up if it is non-integer
    h = (input_shape[0][2] - kernel_shape[0] + pads[0] + pads[2])/strides[0] + 1
    if h > int(h):
        output_shape_h = int(h) + 1
        pads[2] += 1
    else:
        output_shape_h = int(h)

    w = (input_shape[0][3] - kernel_shape[1] + pads[1] + pads[3])/strides[1] + 1
    if w > int(w):
        output_shape_w = int(w) + 1
        pads[3] += 1
    else:
        output_shape_w = int(w)

    if kernel_shape[0] == 0:
        output_shape_h,output_shape_w = (1,1)
    if not with_indices:
        output_shape = [[input_shape[0][0],input_shape[0][1],output_shape_h,output_shape_w]]
    else:
        output_shape = [[input_shape[0][0],input_shape[0][1],output_shape_h,output_shape_w], [input_shape[0][0],input_shape[0][1],output_shape_h,output_shape_w]]
    return output_shape

# Build node
def createPooling(layer,nodename,inname,outname,input_shape):
    dict = getPoolingAttri(layer)
    with_indices = True if len(outname) == 2 else False
    output_shape = getPoolingOutShape(input_shape,layer, dict, with_indices=with_indices)

    # Judgment is the type of pooling, maximum pooling, average pooling
    if layer.pooling_param.pool == 0:
        if layer.pooling_param.global_pooling == True:
            node = Node.c2oNode(layer, nodename, "GlobalMaxPool", inname, outname, input_shape, output_shape, dict={})
        else:
            node = Node.c2oNode(layer, nodename, "MaxPool", inname, outname, input_shape, output_shape, dict=dict)
    elif layer.pooling_param.pool == 1:
        if layer.pooling_param.global_pooling == True:
         
            node = Node.c2oNode(layer, nodename, "GlobalAveragePool", inname, outname, input_shape, output_shape, dict={})
        else:
            node = Node.c2oNode(layer, nodename, "AveragePool", inname, outname, input_shape, output_shape, dict=dict)
    # Layers [i] .pooling_param.pool == 2 is random pooling
    return node
