import src.c2oObject as Node
##---------------------------------------------Softmax--------------------------------------------------------------##
# Get hyperparameters
def getSoftmaxAttri(layer):
    axis = layer.softmax_param.axis
    # Hyperparameter dictionary
    dict = {"axis": axis}
    return dict

# Calculate the output dimension
def getSoftmaxOutShape(input_shape):
    # Calculate the output dimensionoutput_shape
    output_shape = input_shape # same as input dimension
    return output_shape

# Build node
def createSoftmax(layer, nodename, inname, outname, input_shape):
    dict = getSoftmaxAttri(layer)
    output_shape = getSoftmaxOutShape(input_shape)
    # Build node
    node = Node.c2oNode(layer, nodename, "Softmax", inname, outname, input_shape, output_shape, dict)
    print(nodename, " node construction completed")
    return node