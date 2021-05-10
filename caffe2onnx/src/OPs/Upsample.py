import src.c2oObject as Node
import numpy as np
##----------------------------------------------Upsample------------------------------------------------------##
# Get hyperparameters
def getUpsampleAttri(layer):
    # scale = layer.upsample_param.scale
    # scales = [1.0,1.0,scale,scale]
    # dict = {"scales":scales,"mode":"nearest"} # upsample put scales into the parameters
    # dict = {"width_scale": scale,"height_scale":scale, "mode": "nearest"} # when OpenVINO reads onnx, it requires width_scale and height_scale
    dict = {"mode": "nearest"}

    return dict

def getUpsampleOutShape(input_shape,layer):
    scale = layer.upsample_param.scale
    scales = [1.0,1.0,scale,scale]
    output_shape = [np.multiply(np.array(scales,dtype=np.int),np.array(input_shape[0])).tolist()]
    return output_shape

def createUpsample(layer, nodename, inname, outname, input_shape):
    dict = getUpsampleAttri(layer)
    output_shape = getUpsampleOutShape(input_shape,layer)

    #print(output_shape)
    node = Node.c2oNode(layer, nodename, "Upsample", inname, outname, input_shape, output_shape, dict)
    print(nodename, " node construction completed")
    return node