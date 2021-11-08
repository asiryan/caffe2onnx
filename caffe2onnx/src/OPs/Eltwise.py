import caffe2onnx.src.c2oObject as Node
##-------------------------------------------------Eltwise----------------------------------------------------------##
def createEltwise(layer, nodename, inname, outname, input_shape):
    # Determine the operator type
    if layer.eltwise_param.operation == 0:
        node = __createMul(layer, nodename, inname, outname, input_shape) # Multiply by element

    elif layer.eltwise_param.operation == 1:
        node = __createAdd(layer, nodename, inname, outname, input_shape) # Add by element

    elif layer.eltwise_param.operation == 2:
        node = __createMax(layer, nodename, inname, outname, input_shape) # Find the maximum value by element

    return node

##----------------------------------------------Mul layer, corresponding to Prod-----------------------------------------------##
def __createMul(layer, nodename, inname, outname, input_shape):
    output_shape = input_shape[0]
    node = Node.c2oNode(layer, nodename, "Mul", inname, outname, input_shape, output_shape)
    return node

##---------------------Add layer, which may be the addition of two intermediate layer outputs, or it may be an output plus a bias------------------------##
def __createAdd(layer, nodename, inname, outname, input_shape):
    output_shape = [input_shape[0]]
    node = Node.c2oNode(layer, nodename, "Add", inname, outname, input_shape, output_shape)
    return node

##----------------------------------------------Max-------------------------------------------------------------##
def __createMax(layer, nodename, inname, outname, input_shape):
    output_shape = input_shape
    node = Node.c2oNode(layer, nodename, "Max", inname, outname, input_shape, output_shape)
    return node