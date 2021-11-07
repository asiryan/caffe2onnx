import caffe2onnx.src.c2oObject as Node

##-----------------------------BatchNormalization layer = BatchNorm + Scale-------------------------------------##
# Get hyperparameters
def getBNAttri(layer):
    # Hyperparameter dictionary
    
    dict = {"epsilon": layer.batch_norm_param.eps,  # Slip coefficient
            "momentum": layer.batch_norm_param.moving_average_fraction
            }
    return dict

# Calculate the output dimension
def getBNOutShape(input_shape):
    output_shape = input_shape
    return output_shape

# Build node
def createBN(layer, nodename, inname, outname, input_shape):
    dict = getBNAttri(layer)
    # Calculate output_shape, the output dimension is equal to the input dimension
    output_shape = getBNOutShape(input_shape)

    # Build node
    node = Node.c2oNode(layer, nodename, "BatchNormalization", inname, outname, input_shape, output_shape,dict)
    return node
