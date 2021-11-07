import copy
import numpy as np
from onnx import helper

from . import OPs as op
from .c2oObject import *
from .op_layer_info import *


class Caffe2Onnx():
    def __init__(self,net,model,onnxname):
        # Initialize a c2oGraph object
        self.onnxmodel = c2oGraph(onnxname)
        # Network and parameters
        self._NetLayer = self.__getNetLayer(net)
        self._ModelLayer = self.__getModelLayer(model)

        # Model input name and input dimension
        self.model_input_name = []
        self.model_input_shape = []

        # Node list
        self.__n = 0
        self.NodeList = []

        # Get layer list
        LayerList = self.__addInputsTVIandGetLayerList(net)
        self.__getNodeList(LayerList)
        self.__addOutputsTVIandValueInfo()

    # Get the network layer
    def __getNetLayer(self,net):
        if len(net.layer)==0 and len(net.layers)!=0:
            return net.layers
        elif len(net.layer)!=0 and len(net.layers)==0:
            return net.layer
        else:
            print("prototxt layer error")
            return -1

    # Get parameter layer
    def __getModelLayer(self,model):
        if len(model.layer) == 0 and len(model.layers) != 0:
            return model.layers
        elif len(model.layer) != 0 and len(model.layers) == 0:
            return model.layer
        else:
            print("caffemodel layer error")
            return -1

    # Add model input information to Inputs and get a list of subsequent layers
    def __addInputsTVIandGetLayerList(self,net):
        # If the type of the first layer is Input, and no net.input exists
        if net.input == [] and self._NetLayer[0].type == "Input":
            layer_list = []
            # Considering that the entire network will have multiple inputs
            for lay in self._NetLayer:
                if lay.type == "Input":
                    in_tvi = helper.make_tensor_value_info(lay.name+"_input", TensorProto.FLOAT, lay.input_param.shape[0].dim)
                    self.model_input_name.append(lay.name+"_input")
                    self.model_input_shape.append(lay.input_param.shape[0].dim)
                    self.onnxmodel.addInputsTVI(in_tvi)
                    #print("add model input information")
                else:
                    layer_list.append(lay)
            return layer_list

        # If net.input exists
        elif net.input !=[]:
            if bool(net.input_dim):
                input_dim = net.input_dim
            elif bool(net.input_shape):
                input_dim = net.input_shape[0].dim
            else:
                raise RuntimeError("Input shape missing!")

            in_tvi = helper.make_tensor_value_info("input", TensorProto.FLOAT, input_dim)
            self.model_input_name.append("input")
            self.model_input_shape.append(input_dim)
            self.onnxmodel.addInputsTVI(in_tvi)
            print("add model input information")
            return self._NetLayer

        # None of the above situations, then the caffe model has no input, there is a problem
        else:
            raise ValueError("the caffe model has no input")


    # Get the parameter shape of layer
    def __getParamsShapeandData(self, layer):
        ParamShape = []
        ParamData = []
        # According to the layer name, find out the parameters in the corresponding caffemodel
        for model_layer in self._ModelLayer:
            if layer.name == model_layer.name:
                Params = copy.deepcopy(model_layer.blobs)
                ParamShape = [p.shape.dim for p in Params]
                ParamData = [p.data for p in Params]
                if layer.type == "BatchNorm" or layer.type == "BN": 
                    if len(ParamShape) == 3:  
                        # If it is a bn layer, the sliding coefficient of the last layer is not used
                        ParamShape = ParamShape[:-1]
                        ParamData = ParamData[:-1]
                    elif len(ParamShape) == 2 and len(ParamShape[0]) != 1:
                        ParamShape = [[ParamShape[0][1]], [ParamShape[1][1]]]
                        ParamData = ParamData
        return ParamShape, ParamData



    # Add parameters to Inputs and generate tensor storage data
    def __addInputsTVIfromParams(self, layer, ParamName, ParamType):
        ParamShape = []
        ParamData = []
        # Find out the parameters in the corresponding caffemodel based on the layer name
        for model_layer in self._ModelLayer:
            if layer.name == model_layer.name:
                Params = copy.deepcopy(model_layer.blobs)
                ParamShape = [p.shape.dim for p in Params]
                ParamData = [p.data for p in Params]
                if layer.type == "BatchNorm" or layer.type == "BN":
                    if len(ParamShape) == 3:
                        # If it is bn layer and params is [mean, var, s], you need to divide mean and var by sliding coefficient s
                        ParamShape = ParamShape[:-1]
                        ParamData = [[q/(Params[-1].data[0]) for q in p.data] if i==0 else [q/(Params[-1].data[0] + 1e-5) for q in p.data] for i,p in enumerate(Params[:-1])]  # with s
                    elif len(ParamShape) == 2 and len(ParamShape[0]) == 4:
                        ParamShape = [[ParamShape[0][1]], [ParamShape[1][1]]]
                        ParamData = [[q/1. for q in p.data] if i==0 else [q/(1. + 1e-5) for q in p.data] for i,p in enumerate(Params)]

                # comment it for tvm because tvm use broadcast at prelu layer
                elif layer.type == "PReLU":
                    ParamShape = [[ParamShape[0][0], 1, 1]]
                break
        
        # Judge whether there is Param
        if ParamShape != []:
            ParamName = ParamName[0:len(ParamShape)]
            ParamType = ParamType[0:len(ParamShape)]
            for i in range(len(ParamShape)):
                ParamName[i] = layer.name+ParamName[i]
                p_tvi = helper.make_tensor_value_info(ParamName[i], ParamType[i], ParamShape[i])
                p_t = helper.make_tensor(ParamName[i],ParamType[i],ParamShape[i],ParamData[i])
                self.onnxmodel.addInputsTVI(p_tvi)
                self.onnxmodel.addInitTensor(p_t)
                #print("add parameters " + ParamName[i] + " input information and tensor data")
        if layer.type == "BatchNorm" or layer.type == "BN" or layer.type == "Scale":
            return ParamName, ParamShape
        return ParamName

    # Manually add parameters to the input information and generate tensor storage data
    def __addInputsTVIfromMannul(self,layer,ParamName,ParamType,ParamShape,ParamData):
        Param_Name = copy.deepcopy(ParamName)
        for i in range(len(ParamShape)):
            Param_Name[i] = layer.name + ParamName[i]
            p_tvi = helper.make_tensor_value_info(Param_Name[i], ParamType[i], ParamShape[i])
            p_t = helper.make_tensor(Param_Name[i], ParamType[i], ParamShape[i], ParamData[i])
            self.onnxmodel.addInputsTVI(p_tvi)
            self.onnxmodel.addInitTensor(p_t)
            #print("add parameters " + Param_Name[i] + " input information and tensor data")
        return Param_Name


    # Get the output name of the previous layer (that is, the input of the current layer)
    def __getLastLayerOutNameAndShape(self,layer):
        outname = []
        outshape = []

        # If the node list is empty, or the bottom of the current layer is in input_name, the input of the previous layer must be Input
        if self.NodeList == []:
            outname += self.model_input_name
            outshape += self.model_input_shape

        else:
            for i in range(len(layer.bottom)):
                for j in range(len(self.model_input_name)):
                    if layer.bottom[i] + '_input' == self.model_input_name[j]:
                        outname.append(self.model_input_name[j])
                        outshape.append(self.model_input_shape[j])

                # Because prototxt has the same name as top and bottom, but layer.bottom can only correspond to one node, so for each layer.bottom, 
                # find the last node with the same name as the upper layer node
                name = None
                shape = None
                for node in self.NodeList:
                    for j in range(len(node.top) if node.node.op_type != "MaxPool" else 1):   # comment if statement for original maxpool and maxunpool
                        if layer.bottom[i] == node.top[j]:
                            name = node.outputs_name[j]
                            shape = node.outputs_shape[j]
                
                if name:
                    outname.append(name)
                    outshape.append(shape)

        try:
            assert outname, "failed at layer %s, layer's bottom not detected ... "%(layer.name)
        except:
            print("failed at layer %s, layer's bottom not detected ... "%(layer.name))

        return outname, outshape

    # Get the output name of the current layer, ie layername + "_ Y"
    def __getCurrentLayerOutName(self,layer):
        # return [layer.name+"_Y"]
        # Consider the situation with multiple outputs
        if layer.top == layer.bottom and len(layer.top) == 1:
            return [layer.name+"_Y"]
        
        return [out+"_Y" for out in layer.top]



    def __getNodeList(self,Layers):
        for i in range(len(Layers)):
            print("convert layer: " + Layers[i].name)
            # Convolution
            if Layers[i].type == "Convolution" or Layers[i].type == Layer_CONVOLUTION:
                # 1. Get node input name, input dimension, output name, node name
                if Layers[i].name == "conv4_3_norm_mbox_loc":
                    import ipdb; ipdb.set_trace()
                inname, input_shape = self.__getLastLayerOutNameAndShape(Layers[i])
                outname = self.__getCurrentLayerOutName(Layers[i])
                nodename = Layers[i].name

                # 2. Generate the node parameter tensor value info, and get the node parameter name, add the parameter name to the node input name list
                conv_pname = self.__addInputsTVIfromParams(Layers[i],op_pname["Conv"],op_ptype["Conv"])
                inname.extend(conv_pname)

                # 3. Build conv_node
                conv_node = op.createConv(Layers[i],nodename,inname,outname,input_shape)

                # 4. Add node to node list
                self.NodeList.append(conv_node)
                self.__n += 1

            # BatchNorm + Scale
            elif Layers[i].type == "BatchNorm" or Layers[i].type == "BN":
                # 1. Get node input name, input dimension, output name, node name
                inname, input_shape = self.__getLastLayerOutNameAndShape(Layers[i]) # Get input name list and input shape
                outname = self.__getCurrentLayerOutName(Layers[i])  # Get the output name list
                nodename = Layers[i].name

                # 2. Generate the node parameter tensor value info, and get the node parameter name, add the parameter name to the node input name list
                if i < len(Layers) - 1 and Layers[i+1].type == "Scale":
                    scale_pname, scale_pshape = self.__addInputsTVIfromParams(Layers[i + 1], op_pname["Scale"], op_ptype["Scale"])
                    bn_pname, bn_pshape = self.__addInputsTVIfromParams(Layers[i], op_pname["BatchNorm"], op_ptype["BatchNorm"])
                    assert bn_pshape == scale_pshape, "BatchNorm and Scale params should share the same shape"
                    inname.extend(scale_pname)
                    inname.extend(bn_pname)

                else:
                    bn_pshape, _ = self.__getParamsShapeandData(Layers[i])
                    custom_params = [np.ones(shape=bn_pshape[0], dtype=np.float), 0.001 + np.zeros(shape=bn_pshape[1], dtype=np.float)]
                    scale_pname = self.__addInputsTVIfromMannul(Layers[i], op_pname["Scale"], op_ptype["Scale"], bn_pshape, custom_params)
                    bn_pname, bn_pshape = self.__addInputsTVIfromParams(Layers[i], op_pname["BatchNorm"], op_ptype["BatchNorm"])
                    inname.extend(scale_pname)
                    inname.extend(bn_pname)


                # 3. Build bn_node
                bn_node = op.createBN(Layers[i], nodename, inname, outname, input_shape)

                # 4. Add node to node list
                self.NodeList.append(bn_node)
                self.__n += 1

            # Pooling
            elif Layers[i].type == "Pooling" or Layers[i].type == Layer_POOLING:
                # 1. Get node input name, input dimension, output name, node name
                inname,input_shape = self.__getLastLayerOutNameAndShape(Layers[i]) # Get input name list and input shape
                outname = self.__getCurrentLayerOutName(Layers[i]) # Get the output name list
                nodename = Layers[i].name

                # 2. Build pool_node
                pool_node = op.createPooling(Layers[i], nodename, inname, outname, input_shape)

                # 3. Add nodes to the node list
                self.NodeList.append(pool_node)
                self.__n += 1


            # MaxUnPool
            elif Layers[i].type == "MaxUnpool":
                # 1. Get node input name, input dimension, output name, node name
                inname, input_shape = self.__getLastLayerOutNameAndShape(Layers[i]) # Get input name list and input shape
                outname = self.__getCurrentLayerOutName(Layers[i]) # Get the output name list
                nodename = Layers[i].name

                # 2. Build unpool_node
                unpool_node = op.createUnPooling(Layers[i], nodename, inname, outname, input_shape)

                # 3. Add nodes to the node list
                self.NodeList.append(unpool_node)
                self.__n += 1


            # Eltwise
            elif Layers[i].type == "Eltwise" or Layers[i].type == Layer_ELTWISE:
                # 1. Get node input name, input dimension, output name, node name
                inname,input_shape = self.__getLastLayerOutNameAndShape(Layers[i]) # Get input name list and input shape
                outname = self.__getCurrentLayerOutName(Layers[i]) # Get the output name list
                nodename = Layers[i].name

                # 2. Buildeltwise_node
                eltwise_node = op.createEltwise(Layers[i], nodename, inname, outname, input_shape)

                # 3. Add nodes to the node list
                self.NodeList.append(eltwise_node)
                self.__n += 1

            # Softmax
            elif Layers[i].type == "Softmax" or Layers[i].type == Layer_SOFTMAX:
                # 1. Get node input name, input dimension, output name, node name
                inname,input_shape = self.__getLastLayerOutNameAndShape(Layers[i]) # Get input name list and input shape
                outname = self.__getCurrentLayerOutName(Layers[i]) # Get the output name list
                nodename = Layers[i].name

                # 2. Buildsoftmax_node
                softmax_node = op.createSoftmax(Layers[i],nodename, inname, outname, input_shape)

                # 3. Add nodes to the node list
                self.NodeList.append(softmax_node)
                self.__n += 1

            # Relu
            elif Layers[i].type == "ReLU" or Layers[i].type == Layer_RELU:
                # 1. Get node input name, input dimension, output name, node name
                inname,input_shape = self.__getLastLayerOutNameAndShape(Layers[i]) # Get input name list and input shape
                outname = self.__getCurrentLayerOutName(Layers[i]) # Get the output name list
                nodename = Layers[i].name

                # 2. Buildrelu_node
                relu_node = op.createRelu(Layers[i], nodename, inname, outname, input_shape)

                # 3. Add nodes to the node list
                self.NodeList.append(relu_node)
                self.__n += 1

            # LRN
            elif Layers[i].type == "LRN" or Layers[i].type == Layer_LRN:
                # 1. Get node input name, input dimension, output name, node name
                inname,input_shape = self.__getLastLayerOutNameAndShape(Layers[i])
                outname = self.__getCurrentLayerOutName(Layers[i])
                nodename = Layers[i].name

                # 2. BuildLRN_node
                LRN_node = op.createLRN(Layers[i],nodename, inname, outname, input_shape)

                # 3. Add nodes to the node list
                self.NodeList.append(LRN_node)
                self.__n += 1

            # Dropout
            elif Layers[i].type == "Dropout" or Layers[i].type == Layer_DROPOUT:
                # 1. Get node input name, input dimension, output name, node name
                inname,input_shape = self.__getLastLayerOutNameAndShape(Layers[i])
                outname = self.__getCurrentLayerOutName(Layers[i])
                nodename = Layers[i].name

                # 2. BuildDropout_node
                Dropout_node = op.createDropout(Layers[i], nodename, inname, outname, input_shape)

                # 3. Add nodes to the node list
                self.NodeList.append(Dropout_node)
                self.__n += 1


            # Upsample
            elif Layers[i].type == "Upsample" or Layers[i].type == Layer_UPSAMPLE:
                # 1. Get node input name, input dimension, output name, node name
                inname, input_shape = self.__getLastLayerOutNameAndShape(Layers[i])
                outname = self.__getCurrentLayerOutName(Layers[i])
                nodename = Layers[i].name

                # 2. Generate the node parameter tensor value info, and get the node parameter name, add the parameter name to the node input name list
                paramshape = [[4, 1]]
                paramdata = [[1.0, 1.0, Layers[i].upsample_param.scale, Layers[i].upsample_param.scale]]
                pname = self.__addInputsTVIfromMannul(Layers[i],op_pname["Upsample"],op_ptype["Upsample"],paramshape,paramdata)
                inname.extend(pname)

                # 3. Build Upsample_node
                Upsample_node = op.createUpsample(Layers[i], nodename, inname, outname, input_shape)

                # 4. Add node to node list
                self.NodeList.append(Upsample_node)
                self.__n += 1

            # Concat
            elif Layers[i].type == "Concat" or Layers[i].type == Layer_CONCAT:
                # 1. Get node input name, input dimension, output name, node name
                inname,input_shape = self.__getLastLayerOutNameAndShape(Layers[i])
                outname = self.__getCurrentLayerOutName(Layers[i])
                nodename = Layers[i].name

                # 2. BuildConcat_node
                Concat_node = op.createConcat(Layers[i], nodename, inname, outname, input_shape)

                # 3. Add nodes to the node list
                self.NodeList.append(Concat_node)
                self.__n += 1

            # PRelu
            elif Layers[i].type == "PReLU":
                # 1. Get node input name, input dimension, output name, node name
                inname,input_shape = self.__getLastLayerOutNameAndShape(Layers[i])
                outname = self.__getCurrentLayerOutName(Layers[i])
                nodename = Layers[i].name

                # 2. Generate the node parameter tensor value info, and get the node parameter name, add the parameter name to the node input name list
                pname = self.__addInputsTVIfromParams(Layers[i], op_pname["PRelu"], op_ptype["PRelu"])
                inname.extend(pname)

                # 3. Build PRelu_node
                PRelu_node = op.createPRelu(Layers[i], nodename, inname, outname, input_shape)

                # 4. Add node to node list
                self.NodeList.append(PRelu_node)
                self.__n += 1


            # InnerProduct
            # Since there is no fully connected layer in onnx, it needs to be split. There are two methods for splitting (Reshape + Gemm, Reshape + MatMul + Add)
            elif Layers[i].type == "InnerProduct" or Layers[i].type == Layer_INNER_PRODUCT:

                reshape_layer = copy.deepcopy(Layers[i]) # Deep copy
                # 1. Get node input name, input dimension, output name, node name
                reshape_inname, reshape_input_shape = self.__getLastLayerOutNameAndShape(reshape_layer)  # Get reshape input name list and input shape
                reshape_outname = [reshape_layer.name + "_Reshape_Y"]
                reshape_nodename = reshape_layer.name+"_Reshape"

                # 2. Generate the node parameter tensor value info, and get the node parameter name, add the parameter name to the node input name list
                paramshape = [[2]]
                paramdata = op.getReshapeOutShape(Layers[i],reshape_input_shape)
                reshape_pname = self.__addInputsTVIfromMannul(reshape_layer,op_pname["Reshape"],op_ptype["Reshape"],paramshape,paramdata)
                reshape_inname.extend(reshape_pname)

                #3. Build reshape_node
                reshape_node = op.createReshape(reshape_layer,reshape_nodename, reshape_inname, reshape_outname, reshape_input_shape)

                # 4. Add node to node list
                self.NodeList.append(reshape_node)
                self.__n += 1


                # Gemm
                gemm_layer = copy.deepcopy(Layers[i]) # Deep copy
                # 1. Get node input name, input dimension, output name, node name
                gemm_inname = reshape_outname
                gemm_input_shape = self.NodeList[self.__n-1].outputs_shape
                gemm_outname = [gemm_layer.name+"_Gemm_Y"]
                gemm_nodename = gemm_layer.name+"_Gemm"


                # 2. Generate the node parameter tensor value info, and get the node parameter name, add the parameter name to the node input name list
                gemm_pname = self.__addInputsTVIfromParams(gemm_layer,op_pname["InnerProduct"],op_ptype["InnerProduct"])  # Obtain input parameters. For add, blobs [1] does not require bias, so directly obtain blobs [0]
                gemm_inname.extend(gemm_pname)


                #3. Build gemm_node
                matmul_node = op.createGemm(gemm_layer, gemm_nodename, gemm_inname, gemm_outname, gemm_input_shape, gemm_layer.inner_product_param.num_output)

                # 4. Add node to node list
                self.NodeList.append(matmul_node)
                self.__n += 1



            # Deconvolution
            elif Layers[i].type == "Deconvolution":
                # 1. Get node input name, input dimension, output name, node name
                inname, input_shape = self.__getLastLayerOutNameAndShape(Layers[i])
                outname = self.__getCurrentLayerOutName(Layers[i])
                nodename = Layers[i].name

                # 2. Generate the node parameter tensor value info, and get the node parameter name, add the parameter name to the node input name list
                conv_pname = self.__addInputsTVIfromParams(Layers[i], op_pname["ConvTranspose"], op_ptype["ConvTranspose"])
                inname.extend(conv_pname)

                #3. Build conv_node
                conv_node = op.createConvTranspose(Layers[i], nodename, inname, outname, input_shape)
                if self.debug:
                    self.__print_debug_info(nodename, inname, outname, input_shape, conv_node.outputs_shape)

                # 4. Add node to node list
                self.NodeList.append(conv_node)
                self.__n += 1



    # Determine whether the current node is an output node
    def judgeoutput(self,current_node,nodelist):
        for outname in current_node.outputs_name:
            for node in nodelist:
                if outname in node.inputs_name:
                    return False
        return True

    # Add model output information and intermediate node information
    def __addOutputsTVIandValueInfo(self):
        for i in range(len(self.NodeList)):
            if self.judgeoutput(self.NodeList[i],self.NodeList):# Build output node information
                lastnode = self.NodeList[i]
                for j in range(len(lastnode.outputs_shape)):
                    output_tvi = helper.make_tensor_value_info(lastnode.outputs_name[j], TensorProto.FLOAT,lastnode.outputs_shape[j])
                    self.onnxmodel.addOutputsTVI(output_tvi)
            else:# Build intermediate node information
                innernode = self.NodeList[i]
                for k in range(len(innernode.outputs_shape)):
                    hid_out_tvi = helper.make_tensor_value_info(innernode.outputs_name[k], TensorProto.FLOAT,innernode.outputs_shape[k])
                    self.onnxmodel.addValueInfoTVI(hid_out_tvi)
        #print("add model output information and model intermediate output information")

    # Create a model
    def createOnnxModel(self):
        node_def = [Node.node for Node in self.NodeList]
        graph_def = helper.make_graph(
            node_def,
            self.onnxmodel.name,
            self.onnxmodel.in_tvi,
            self.onnxmodel.out_tvi,
            self.onnxmodel.init_t,
            value_info=self.onnxmodel.hidden_out_tvi
        )
        model_def = helper.make_model(graph_def, producer_name='caffe')
        print("converting caffe model to onnx model completed successfully")
        return model_def

