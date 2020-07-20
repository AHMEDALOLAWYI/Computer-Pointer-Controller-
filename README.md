# Computer Pointer Controller
In this project three different models were used from OpenVINO model zoo namely the face detection, Head Pose Estimation and Facial landmark Detection. Using these models it will be possible to contorll the mouse pointer based on the estimated pose of the head and the direction the eyes are moving into . The application work by uplaoding a video or using the camera as an input and then it starts moving the pointer contorller based on estimated head pose and eyes direction. The following diagram explains the project flow that was followed to run the application and obtainn results. 
![](Flow%20of%20project.JPG)



## Project Set Up and Installation
*TODO:* Explain the setup procedures to run your project. For instance, this can include your project directory structure, the models you need to download and where to place them etc. Also include details about how to install the dependencies your project requires.
The project will run by satsifiying the following requirments: 
The OpenVINO toolkit is used to run the application and Intel provides detail documenation about the tool on the website as well as many video on youtube that are uplaoded regularly. Depending on the device being used there is an installation procedure and setup : 

https://docs.openvinotoolkit.org/latest/openvino_docs_install_guides_installing_openvino_linux.html

In case of my application running I downloaded OpenVINO on a virtualbox that has built in ubuntu along with 


## Demo
*TODO:* Explain how to run a basic demo of your model.

## Documentation
The models used are pretrained models from the model zoo and each model used for this project has its own documenation which will refer it in a separete link below the model name:

1.Gaze Estimation Model 
https://docs.openvinotoolkit.org/latest/omz_models_intel_gaze_estimation_adas_0002_description_gaze_estimation_adas_0002.html

2.Head Pose Estimation Model 
https://docs.openvinotoolkit.org/latest/omz_models_intel_head_pose_estimation_adas_0001_description_head_pose_estimation_adas_0001.html

3.Landmark Detection Model 
https://docs.openvinotoolkit.org/latest/omz_models_intel_landmarks_regression_retail_0009_description_landmarks_regression_retail_0009.html

4.Face Detection Mdoel 
https://docs.openvinotoolkit.org/latest/omz_models_intel_face_detection_adas_binary_0001_description_face_detection_adas_binary_0001.html

## Benchmarks
*TODO:* Include the benchmark results of running your model on multiple hardwares and multiple model precisions. Your benchmarks can include: model loading time, input/output processing time, model inference time etc.

## Results
*TODO:* Discuss the benchmark results and explain why you are getting the results you are getting. For instance, explain why there is difference in inference time for FP32, FP16 and INT8 models.

## Stand Out Suggestions
This is where you can provide information about the stand out suggestions that you have attempted.

### Async Inference
If you have used Async Inference in your code, benchmark the results and explain its effects on power and performance of your project.

### Edge Cases
There will be certain situations that will break your inference flow. For instance, lighting changes or multiple people in the frame. Explain some of the edge cases you encountered in your project and how you solved them to make your project more robust.
