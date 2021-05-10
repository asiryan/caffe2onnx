import caffe2onnx.src.c2oObject as Node
##----------------------------------------------------Dropout-------------------------------------------------------##
# Get hyperparameters
def getDropoutAttri(layer):
    # Drop ratio
    ratio = layer.dropout_param.dropout_ratio
    # Hyperparameter dictionary
    dict = {"ratio":ratio}
    return dict

def getDropoutOutShape(input_shape):
    #  Calculate the output dimensionoutput_shape
    output_shape = input_shape  # Same as input dimension
    return output_shape

# Build node
def createDropout(layer, nodename, inname, outname, input_shape):
    dict = getDropoutAttri(layer)
    output_shape = getDropoutOutShape(input_shape)
    #  Build node
    node = Node.c2oNode(layer, nodename, "Dropout", inname, outname, input_shape, output_shape, dict=dict)
    print(nodename, " node construction completed")
    return node