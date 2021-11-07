import caffe2onnx.src.c2oObject as Node
##--------------------------------------------------Relu------------------------------------------------------------##
# Get hyperparameters
def getReluAttri(layer):
    dict = {}
    if layer.relu_param.negative_slope != 0:
        dict = {"alpha":layer.relu_param.negative_slope}
    return dict
# Calculate the output dimension
def getReluOutShape(input_shape):
    # Get output_shape
    output_shape = input_shape
    return output_shape
# Build node
def createRelu(layer,nodename,inname,outname,input_shape):
    dict = getReluAttri(layer)
    output_shape = getReluOutShape(input_shape)

    if dict == {}:
        node = Node.c2oNode(layer, nodename, "Relu", inname, outname, input_shape, output_shape)
    else:
        node = Node.c2oNode(layer, nodename, "LeakyRelu", inname, outname, input_shape, output_shape, dict=dict)

    return node