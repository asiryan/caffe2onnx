import src.c2oObject as Node
##-----------------------------------------------------Gemm-------------------------------------------------------##
# Get hyperparameters
def getGemmAttri(layer):
    # Hyperparameter dictionary
    dict = {"alpha": 1.0,
            "beta": 1.0,
            "transA": 0,
            "transB": 1}
    return dict

# Calculate the output dimension
def getGemmOutShape(input_shape,num_output):
    output_shape = [[input_shape[0][0], num_output]]
    return output_shape

# Build node
def createGemm(layer, nodename, inname, outname, input_shape, num_output):
    dict = getGemmAttri(layer)
    output_shape = getGemmOutShape(input_shape,num_output)
    # Build node
    node = Node.c2oNode(layer, nodename, "Gemm", inname, outname, input_shape, output_shape, dict)
    print(nodename, " node construction completed")
    return node