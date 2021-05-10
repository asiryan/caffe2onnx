import src.c2oObject as Node
##---------------------------------------------Concat-----------------------------------------------------------##
# Get hyperparameters
def getConcatAttri(layer):
    axis = layer.concat_param.axis
    dict = {"axis":axis}
    return dict

# Calculate the output dimension
def getConcatOutShape(input_shape):
    n,c,w,h = input_shape[0][0],0,input_shape[0][2],input_shape[0][3]
    for i in range(len(input_shape)):
        c = c+input_shape[i][1]
    output_shape = [[n,c,w,h]]
    return output_shape

# Build node
def createConcat(layer, nodename, inname, outname, input_shape):
    dict = getConcatAttri(layer)
    output_shape = getConcatOutShape(input_shape)

    node = Node.c2oNode(layer, nodename, "Concat", inname, outname, input_shape, output_shape, dict)
    print(nodename, " node construction completed")
    return node