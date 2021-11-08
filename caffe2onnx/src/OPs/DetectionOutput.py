import onnx
from typing import *
from onnx import helper
from typing import *
import ctypes
import caffe2onnx.src.c2oObject as Node


def create_attribuates(layer) -> Dict:
    detection_output_param = layer.detection_output_param
    num_classes  = detection_output_param.num_classes
    share_location        = 1 if detection_output_param.share_location else 0
    background_label_id    = detection_output_param.background_label_id
    # NonMaximumSuppressionParameter
    nms_threshold = detection_output_param.nms_param.nms_threshold
    top_k = detection_output_param.nms_param.top_k
    eta = detection_output_param.nms_param.eta

    code_type              = detection_output_param.code_type
    variance_encoded_in_target = 1 if detection_output_param.variance_encoded_in_target else 0
    keep_top_k  = detection_output_param.keep_top_k
    confidence_threshold = detection_output_param.confidence_threshold
    visualize = 1 if detection_output_param.visualize else 0
    visualize_threshold = detection_output_param.visualize_threshold
    save_file = detection_output_param.save_file



    # TODO: SaveOutputParameter
    # save_output_param = detection_output_param.save_output_param
    # output_directory: str = save_output_param.output_directory
    # output_name_prefix: str = save_output_param.output_name_prefix
    # output_format: str = save_output_param.output_format
    # label_map_file: str = save_output_param.label_map_file
    # name_size_file: str = save_output_param.name_size_file
    # num_test_image: int = save_output_param.num_test_image



    attributes = {
        'num_classes'            : num_classes,
        'share_location'       : share_location,
        'background_label_id'  : background_label_id,
        'nms_threshold'        : nms_threshold,
        'top_k'                : top_k,
        'eta'                  : eta,
        'code_type'            : code_type,
        'variance_encoded_in_target' : variance_encoded_in_target,
        'keep_top_k'           : keep_top_k,
        'confidence_threshold' : confidence_threshold,
        'visualize'            : visualize,
        'visualize_threshold'  : visualize_threshold,
        'save_file'            : save_file
        }
    return attributes


def create_detection_output(layer,
                            node_name: str,
                            inputs_name: List[str],
                            outputs_name: List[str],
                            inputs_shape: List, ) -> onnx.NodeProto:

    attributes = create_attribuates(layer)

    outputs_shape = [[1, 1, 1, 7]]

    node = Node.c2oNode(layer, node_name, "DetectionOutput",
                        inputs_name, outputs_name,
                        inputs_shape, outputs_shape,
                        attributes)
    return node
