'''
Class to manipulate models used in gaze pointer controller.
'''
import os
import sys
import helpers 
from openvino.inference_engine import IENetwork, IECore

class GenericModel:
    '''
    Class for controlling similar model characteristics.
    '''
    def __init__(self, device):
        self.device = device
        self.net = None
        self.plugin = None
        self.input_blob = None
        self.out_blob = None
        self.net_plugin = None
        self.infer_request_handle = None
        self.outputs = None


    def load_model(self, model, cpu_extension=None, plugin=None):
        # Obtain model files path:
        model_xml = model
        model_bin = os.path.splitext(model_xml)[0] + ".bin"

        if not plugin:
            self.plugin = IECore()
        else:
            self.plugin = plugin

        if cpu_extension and 'CPU' in self.device:
            self.plugin.add_extension(cpu_extension, "CPU")


        self.net = IENetwork(model=model_xml, weights=model_bin)



        # If applicable, add a CPU extension to self.plugin
        if "CPU" in self.device:
            supported_layers = self.plugin.query_network(self.net, "CPU")
            not_supported_layers = [layer for layer in self.net.layers.keys() if layer not in supported_layers]

            if len(not_supported_layers) != 0:
       
                sys.exit(1)

        # Load the model to the network:
        self.net_plugin = self.plugin.load_network(network=self.net, device_name=self.device)

        # Obtain other relevant information:
        self.input_blob = next(iter(self.net.inputs))
        self.out_blob = next(iter(self.net.outputs))
        return self.plugin

    def predict(self, image, request_id=0):

        preprocessed_image = self.preprocess_input(image)
        self.infer_request_handle = self.net_plugin.start_async(request_id=request_id, 
                                                                inputs={self.input_blob: preprocessed_image})

        return self.net_plugin

    def check_model(self):
        pass

    def preprocess_input(self, image):

        input_shape = self.net.inputs[self.input_blob].shape
        preprocessed_image = helpers.handle_image(image, input_shape[3], input_shape[2])

        return preprocessed_image

    def wait(self, request_id=0):
        status = self.net_plugin.requests[request_id].wait(-1)

        return status

    def get_output(self, request_id=0):

        self.outputs = self.net_plugin.requests[request_id].outputs[self.out_blob]

        return self.outputs



