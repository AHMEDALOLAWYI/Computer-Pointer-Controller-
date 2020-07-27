'''Script to run the application'''


import numpy as np
import helpers
import cv2

from modelfacelandmark import FaceLandmarks
from modelgaze import Gaze 
from modelfacedetector import FaceDetector
from mouse_controller import MouseController
from headposemodel import HeadPose
from argparse import ArgumentParser



def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Video or Image file path. Use CAMERA to use webcam stream")
    
    

    parser.add_argument("-hp", "--headpose_model", required=False, type=str, default=None,
                        help="xam path to file with a trained head pose detector model.")
    parser.add_argument("-g", "--gaze_model", required=False, type=str, default=None,
                        help="xmal path to file with a trained gaze detector model.")
    parser.add_argument("-f", "--facedetector_model", required=False, type=str, default=None,
                        help=" facedetector model model path.")
    parser.add_argument("-lm", "--facelm_model", required=False, type=str, default=None,
                        help="face landmarks detector model path.")
   
    parser.add_argument("-sf", "--show_face", type=bool, default=False,
                        help="Draw face bounding box")

    parser.add_argument("-se", "--show_eyes", type=bool, default=False,
                    help="Flag for the indication that user wants to see eyes bounding boxes")

    parser.add_argument("-sh", "--show_headpose", type=bool, default=False,
                    help="Flag for the indication that user wants to see headpose angles")

    parser.add_argument("-sg", "--show_gaze", type=bool, default=False,
                    help="Flag for the indication that user wants to see gaze direction")


                                                              
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")


    
    return parser


def initiate_models(args):
    facedetector_model = args.facedetector_model
    facelm_model = args.facelm_model
    headpose_model = args.headpose_model
    gaze_model = args.gaze_model

   
    if facedetector_model:
        facedetector = FaceDetector(model_path=facedetector_model, device=args.device)
    else:
        facedetector = FaceDetector(device=args.device)
    
    
    if facelm_model:
        facelm = FaceLandmarks(model_path=facelm_model, device=args.device)
    else:
        facelm = FaceLandmarks(device=args.device)
    
   
    if headpose_model:
        headpose = HeadPose(model_path=headpose_model, device=args.device)
    else:
        headpose = HeadPose(device=args.device)

    if gaze_model:
        gaze = Gaze(model_path=gaze_model, device=args.device)
    else:
        gaze = Gaze()

    return facedetector, facelm, headpose, gaze


def application(args, facedetector, facelm, headpose, gaze):

    pointer_controller = MouseController(precision='high', speed='fast')

    
   
    if args.input != 'CAM':
        try:
            
            in_stream = cv2.VideoCapture(args.input)
            l = int(in_stream.get(cv2.CAP_PROP_FRAME_COUNT))
            webcam = False

            
            if l > 1:
                single_image_mode = False
            else:
                single_image_mode = True

        except:
            print('Not supported image or video file format. Please pass a supported one.')
            exit()

    else:
        in_stream = cv2.VideoCapture(0)
        single_img_mode = False
        webcam = True

    if not single_image_mode:
        count = 0
        while(in_stream.isOpened()):
        
            
            flag, frame = in_stream.read()

            if not flag:
                break

            if count % 25 == 0:

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                
                face_crop, detection = facedetector.get_face_crop(frame, args)

               
                right_eye, left_eye = facelm.get_eyes_coordinates(face_crop)

                
                right_eye_crop, left_eye_crop, right_eye_coords, left_eye_coords = helpers.get_eyes_crops(face_crop, right_eye,left_eye)
               


                
                headpose_angles = headpose.get_headpose_angles(face_crop)


                


                
                (x_movement, y_movement), gaze_vector = gaze.get_gaze(right_eye_crop, left_eye_crop, headpose_angles)


                
                if args.show_face:
                    frame = cv2.rectangle(frame,
                                          (detection[0],detection[1]),
                                          (detection[2],detection[3]), 
                                          color=(0,255,0), 
                                          thickness=5)
                if args.show_headpose:

                    frame = cv2.putText(frame, 'Roll: '+
                                    str(headpose_angles[2])+' '+
                                    'Pitch: '+str(headpose_angles[1])+' '+
                                    'Yaw: '+str(headpose_angles[0]),(15,20),cv2.FONT_HERSHEY_SIMPLEX,0.65,(0,0,0),2)

                if args.show_eyes:

                    right_eye_coords = [right_eye_coords[0]+detection[1], right_eye_coords[1]+detection[1], 
                                right_eye_coords[2]+detection[0], right_eye_coords[3]+detection[0]]

                    left_eye_coords = [left_eye_coords[0]+detection[1], left_eye_coords[1]+detection[1], 
                                left_eye_coords[2]+detection[0], left_eye_coords[3]+detection[0]]
                    frame = cv2.rectangle(frame,
                                            (right_eye_coords[2],right_eye_coords[1]),
                                            (right_eye_coords[3],right_eye_coords[0]), 
                                            color=(255,0,0), 
                                            thickness=5)

                    frame = cv2.rectangle(frame,
                                            (left_eye_coords[2],left_eye_coords[1]),
                                            (left_eye_coords[3],left_eye_coords[0]), 
                                            color=(255,0,0), 
                                            thickness=5)

                if args.show_gaze:
                
                    # Right eye:
                    x_r_eye = int(right_eye[0]*face_crop.shape[1]+detection[0])
                    y_r_eye = int(right_eye[1]*face_crop.shape[0]+detection[1])
                    x_r_shift, y_r_shift = int(x_r_eye+gaze_vector[0]*100), int(y_r_eye-gaze_vector[1]*100)

                    # Left eye:
                    x_l_eye = int(left_eye[0]*face_crop.shape[1]+detection[0])
                    y_l_eye = int(left_eye[1]*face_crop.shape[0]+detection[1])
                    x_l_shift, y_l_shift = int(x_l_eye+gaze_vector[0]*100), int(y_l_eye-gaze_vector[1]*100)

                    frame = cv2.arrowedLine(frame, (x_r_eye, y_r_eye), (x_r_shift, y_r_shift), (0, 255, 0), 4)
                    frame = cv2.arrowedLine(frame, (x_l_eye, y_l_eye), (x_l_shift, y_l_shift), (0, 255, 0), 4)

                

                 
                cv2.namedWindow('Output',cv2.WINDOW_NORMAL)
                cv2.resizeWindow('Output', 800,600)
                cv2.imshow('Output', frame)

                #pointer_controller.move(x_movement,y_movement)
            count = count + 1

        in_stream.release()
        
    cv2.destroyAllWindows()


def main():
    """
    Load the network and parse the output.

    :return: None
    """
   
    args = build_argparser().parse_args()

    
    facedetector, facelm, headpose, gaze = initiate_models(args)

    
    application(args, facedetector, facelm, headpose, gaze)


if __name__ == '__main__':
    main()
