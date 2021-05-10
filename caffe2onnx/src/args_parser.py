import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Convert Caffe model to ONNX.')

    parser.add_argument("--prototxt",
                        type=str,  required=True,
                        help="input .prototxt")

    parser.add_argument("--caffemodel",
                        type=str,
                        required=True,
                        help="input .caffemodel")

    parser.add_argument("--onnx",
                        type=str,
                        required=False,
                        help="output .onnx")

    parser.add_argument("--frozen",
                        type=bool, required=False,
                        help="frozen graph or not")

    args = parser.parse_args()
    return args