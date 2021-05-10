import onnx

def freeze(onnxmodel):
    print("removing not constant initializers from model")
    inputs = onnxmodel.graph.input
    name_to_input = {}
    for input in inputs:
        name_to_input[input.name] = input

    for initializer in onnxmodel.graph.initializer:
        if initializer.name in name_to_input:
            inputs.remove(name_to_input[initializer.name])

    print("frozen graph has been created")
    return
