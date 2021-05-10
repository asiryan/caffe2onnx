import caffe2onnx.src.c2oObject as Node
##-------------------------------------------------LRN-------------------------------------------------------------##
# Get hyperparameters
def getLRNAttri(layer):
    #  Get hyperparameters
    # Size
    size = layer.lrn_param.local_size
    # Alpha
    alpha = layer.lrn_param.alpha
    # Beta
    beta = layer.lrn_param.beta

    # Hyperparameter dictionary
    dict = {"alpha":alpha,
            "beta":beta,
            "bias":1.0,
            "size": size}
    return dict

# Calculate the output dimension
def getLRNOutShape(input_shape):
    #  Calculate the output dimensionoutput_shape
    output_shape = input_shape  # Same as input dimension
    return output_shape

# Build node
def createLRN(layer,nodename, inname,outname,input_shape):
    dict = getLRNAttri(layer)
    output_shape = getLRNOutShape(input_shape)

    # Build node
    node = Node.c2oNode(layer, nodename, "LRN", inname, outname, input_shape, output_shape, dict)
    print(nodename, " node construction completed")
    return node