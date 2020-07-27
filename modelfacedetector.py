'''
Class to manipulate models used in gaze pointer controller.
'''
from openvino.inference_engine import IENetwork, IECore
from model import GenericModel


Face_Model_Path = 'intel/face-detection-adas-0001/FP16/face-detection-adas-0001.xml'

class FaceDetector(GenericModel):

    def __init__(self, model_path=Face_Model_Path, device='CPU'):
        super().__init__(device=device)
        self.load_model(model_path)
    
    def preprocess_output(self):

        return self.outputs[0,0]

    def get_face_crop(self, frame, args):

        thresh = args.prob_threshold

        try:
            self.predict(frame)
            self.wait()
            self.get_output()
            out = self.preprocess_output()
            
        except:
            return None

        if len(out)>0:
            det = []
            for o in out:
                if o[2] > thresh:
                    xminimum = o[3]
                    yminimum = o[4]
                    xmaximum = o[5]
                    ymaximum = o[6]
                    det.append([xminimum, yminimum, xmaximum, ymaximum])

        
        det = det[0]
        w = frame.shape[1]
        h = frame.shape[0]
        det = [int(det[0]*w), int(det[1]*h), int(det[2]*w), int(det[3]*h)]

        return frame[det[1]:det[3], det[0]:det[2]], det


