'''
Class to manipulate models used in gaze pointer controller.
'''
from openvino.inference_engine import IENetwork, IECore
from model import GenericModel

FACE_LANDMARKS_MODEL = 'intel/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009.xml'



class FaceLandmarks(GenericModel):

    def __init__(self, model_path=FACE_LANDMARKS_MODEL, device='CPU'):
        super().__init__(device=device)
        self.load_model(model_path)
    
    def preprocess_output(self):

        return self.outputs[0,:,0,0]

    def get_eyes_coordinates(self, face_crop):

        try:
            self.predict(face_crop)
            self.wait()
            self.get_output()
            out = self.preprocess_output()
        except:
            return None

        if len(out) > 0:
            r_eye = (out[0], out[1])
            l_eye = (out[2], out[3])

        return r_eye, l_eye



