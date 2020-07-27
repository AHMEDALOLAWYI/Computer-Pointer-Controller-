'''
Class to manipulate models used in gaze pointer controller.
'''
from openvino.inference_engine import IENetwork, IECore
from model import GenericModel


HEAD_Model_Path = 'intel/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001.xml'

class HeadPose(GenericModel):

    def __init__(self, model_path=HEAD_Model_Path, device='CPU'):
        super().__init__(device=device)
        self.load_model(model_path)


    def get_output(self, request_id=0):
        self.outputs = self.net_plugin.requests[request_id].outputs

        return self.outputs


    def preprocess_output(self):
        return [self.outputs['angle_y_fc'][0,0], self.outputs['angle_p_fc'][0,0], self.outputs['angle_r_fc'][0,0]]


    
    def get_headpose_angles(self, face_crop):

        try:
            self.predict(face_crop)
            self.wait()
            self.get_output()
            output = self.preprocess_output()
        except:
            return None

        return output



