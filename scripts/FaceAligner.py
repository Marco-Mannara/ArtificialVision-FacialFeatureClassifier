import os 
import cv2
import dlib

import numpy as np

from collections import OrderedDict

'''
# For dlib’s 68-point facial landmark detector:
FACIAL_LANDMARKS_68_IDXS = OrderedDict([
    ("mouth", (48, 68)),
    ("inner_mouth", (60, 68)),
    ("right_eyebrow", (17, 22)),
    ("left_eyebrow", (22, 27)),
    ("right_eye", (36, 42)),
    ("left_eye", (42, 48)),
    ("nose", (27, 36)),
    ("jaw", (0, 17))
])
'''

FACIAL_LANDMARKS_5_IDXS = OrderedDict([
    ("left_eye", (1,2)) ,
    ("right_eye", (3,4)),
    ("nose", (5,5))
])

def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)

    # loop over all facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords


class FaceAligner:
    def __init__(self,  desiredLeftEye=(0.35, 0.35),
                 desiredFaceWidth=256, desiredFaceHeight=None):
        # store the facial landmark predictor, desired output left
        # eye position, and desired output face width + height
        #net = cv2.dnn.readNetFromDarknet(cfg_path, weight_path)
        #self.detector = cv2.dnn_DetectionModel(net)
        #self.detector.setInputParams(scale=1 / 255, size=(416, 416), swapRB=True)
        #self.detector = dlib.cnn_face_detection_model_v1(os.path.join('trained_models', 'mmod_human_face_detector.dat'))
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor( os.path.join('trained_models','shape_predictor_5_face_landmarks.dat'))
        self.desiredLeftEye = desiredLeftEye
        self.desiredFaceWidth = desiredFaceWidth
        self.desiredFaceHeight = desiredFaceHeight

        # if the desired face height is None, set it to be the
        # desired face width (normal behavior)
        if self.desiredFaceHeight is None:
            self.desiredFaceHeight = self.desiredFaceWidth

    def align(self,img):
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray,1)
        faces = [f for f in faces]

        #rect = None 
        #classIds, scores, boxes = self.detector.detect(img, confThreshold=0.6, nmsThreshold=0.4)
        #detection_data = zip(classIds,scores, boxes)
        #data_person = [(id,sc,box) for id,sc,box in detection_data if id == 0]
        
        #if len(data_person) > 0:        
        #    box = data_person[0][2]
        #    rect = dlib.rectangle(left = box[0], right = box[2], top = box[1], bottom = box[3])

       
        if len(faces) > 0:
            # convert the landmark (x, y)-coordinates to a NumPy array
            rect = faces[0]
            shape = self.predictor(gray, rect)
            shape = shape_to_np(shape)

            # extract the left and right eye (x, y)-coordinates
            (lStart, lEnd) = FACIAL_LANDMARKS_5_IDXS["left_eye"]
            (rStart, rEnd) = FACIAL_LANDMARKS_5_IDXS["right_eye"]

            leftEyePts = shape[lStart:lEnd]
            rightEyePts = shape[rStart:rEnd]

            # compute the center of mass for each eye
            leftEyeCenter = leftEyePts.mean(axis=0).astype("int")
            rightEyeCenter = rightEyePts.mean(axis=0).astype("int")

            # compute the angle between the eye centroids
            dY = rightEyeCenter[1] - leftEyeCenter[1]
            dX = rightEyeCenter[0] - leftEyeCenter[0]
            angle = np.degrees(np.arctan2(dY, dX)) - 180

            # compute the desired right eye x-coordinate based on the
            # desired x-coordinate of the left eye
            desiredRightEyeX = 1.0 - self.desiredLeftEye[0]

            # determine the scale of the new resulting image by taking
            # the ratio of the distance between eyes in the *current*
            # image to the ratio of distance between eyes in the
            # *desired* image
            dist = np.sqrt((dX ** 2) + (dY ** 2))
            desiredDist = (desiredRightEyeX - self.desiredLeftEye[0])
            desiredDist *= self.desiredFaceWidth
            scale = desiredDist / dist

            # compute center (x, y)-coordinates (i.e., the median point)
            # between the two eyes in the input image
            eyesCenter = (int((leftEyeCenter[0] + rightEyeCenter[0]) // 2),
                        int((leftEyeCenter[1] + rightEyeCenter[1]) // 2))

            # grab the rotation matrix for rotating and scaling the face
            M = cv2.getRotationMatrix2D(center=eyesCenter, angle=angle, scale=scale)

            # update the translation component of the matrix
            tX = self.desiredFaceWidth * 0.5
            tY = self.desiredFaceHeight * self.desiredLeftEye[1]
            M[0, 2] += (tX - eyesCenter[0])
            M[1, 2] += (tY - eyesCenter[1])

            # apply the affine transformation
            (w, h) = (self.desiredFaceWidth, self.desiredFaceHeight)
            output = cv2.warpAffine(gray, M, (w, h),flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_REPLICATE)
            gray_output = cv2.warpAffine(gray, M,(w,h), flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_REPLICATE)
            # return the aligned face
            return output,gray_output,True
        else:
            return img,gray,False

