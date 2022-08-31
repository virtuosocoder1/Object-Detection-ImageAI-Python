# imported ImageAI a object detection library.
from imageai.Detection import ObjectDetection
import os # imported os

execution_path = os.getcwd() # defining a path variable 

detector = ObjectDetection() #  object Detection Class.
detector.setModelTypeAsRetinaNet() # set the model type to RetinaNet.
detector.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.1.0.h5")) 
detector.loadModel() #load the model into the object detection class.
detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , "image.jpg"), output_image_path=os.path.join(execution_path , "imagenew.jpg"))
#then we called the detection function and parsed in the input image path and the output image path

#iterate over all the results returned by the detector.detectObjectsFromImage.
for eachObject in detections:
    print(eachObject["name"] , " : " , eachObject["percentage_probability"] )