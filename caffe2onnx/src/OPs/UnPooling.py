import numpy as np
import caffe2onnx.src.c2oObject as Node
##-----------------------------------------------------UnPooling--------------------------------------------------##
# Get hyperparameters
def getUnPoolingAttri(layer):
    # pooled core size
    # kernel_shape = np.array([layer.pooling_param.kernel_size]*2).reshape(1,-1)[0].tolist()
    # if layer.pooling_param.kernel_size == []:
    #     kernel_shape = [layer.pooling_param.kernel_h,layer.pooling_param.kernel_w]
    # step size
    # strides = [1, 1]
    # if layer.pooling_param.stride != []:
    #     strides = np.array([layer.pooling_param.stride]*2).reshape(1,-1)[0].tolist()
    # filling
    # pads = [0, 0, 0, 0] default is 0
    # here is the same as the convolution, if there is a pad, set it according to its value
    # if layer.pooling_param.pad != []:
    #     pads = np.array([layer.pooling_param.pad] * 4).reshape(1, -1)[0].tolist()
    # elif layer.pooling_param.pad_h != 0 or layer.pooling_param.pad_w != 0:
    #     pads = [layer.pooling_param.pad_h,layer.pooling_param.pad_w,layer.pooling_param.pad_h,layer.pooling_param.pad_w]

    # Hyperparameter dictionary
    dict = {"kernel_shape": [2, 2],
            "strides": [2, 2],
            "pads": [0, 0, 0, 0]
            }
    return dict

# Calculate the output dimension
def getUnPoolingOutShape(input_shape,layer,dict):
    kernel_shape = dict["kernel_shape"]
    pads = dict["pads"]
    strides = dict["strides"]

    # Calculate the output dimension, as with convolution, round up if it is non-integer
    # h = (input_shape[0][2] - kernel_shape[0] + 2 * pads[0])/strides[0] + 1
    # if h > int(h):
    #     output_shape_h = int(h) + 1
    #     pads = [0,0,1,1]
    # else:
    #     output_shape_h = int(h)
    # output_shape = [[input_shape[0][0],input_shape[0][1],output_shape_h,output_shape_h]]

    output_shape = [[input_shape[0][0], input_shape[0][1], input_shape[0][2]*2, input_shape[0][3]*2]]
    return output_shape

# Build node
def createUnPooling(layer,nodename,inname,outname,input_shape):
    dict = getUnPoolingAttri(layer)
    output_shape = getUnPoolingOutShape(input_shape,layer,dict)

    node = Node.c2oNode(layer, nodename, "MaxUnpool", inname, outname, input_shape, output_shape, dict=dict)

    return node