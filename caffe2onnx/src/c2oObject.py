from onnx import helper


class c2oNode(object):
    def __init__(self,layer,node_name,type,inputs_name,outputs_name,inputs_shape,outputs_shape,dict={}, Flag=False):
        self.node = self.__createNode(type, inputs_name, outputs_name, node_name, dict)

        self.bottom = layer.bottom
        if Flag is True:
            self.top = outputs_name
        else:
            self.top = layer.top
            
        self.inputs_name = inputs_name
        self.outputs_name = outputs_name
        self.inputs_shape = inputs_shape
        self.outputs_shape = outputs_shape

        self.dict = dict

    # Create a node
    def __createNode(self, node_type, in_name, out_name, node_name, dict):
        node_def = helper.make_node(
            node_type,
            in_name,
            out_name,
            node_name,
            **dict,
        )
        return node_def

class c2oGraph():
    def __init__(self,onnxname):
        self.name = onnxname
        self.in_tvi = []            # Store input information
        self.out_tvi = []           # Store output information
        self.init_t = []            # Store the value of the input parameter
        self.hidden_out_tvi = []    # Store intermediate output information

    def addInputsTVI(self,in_tvi):
        self.in_tvi.append(in_tvi)
    def addOutputsTVI(self,out_tvi):
        self.out_tvi.append(out_tvi)
    def addInitTensor(self,init_t):
        self.init_t.append(init_t)
    def addValueInfoTVI(self,vi_tvi):
        self.hidden_out_tvi.append(vi_tvi)
