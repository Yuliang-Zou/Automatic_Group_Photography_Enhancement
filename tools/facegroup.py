import cv2
import numpy as np
import glob

def getFaceData(img):
    # Create the haar cascade
    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    # Read the image
    image = cv2.imread(img)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags = cv2.cv.CV_HAAR_SCALE_IMAGE
     )
    for (x, y, w, h) in faces:
        facedata = image[y:y+h, x:x+w]
    return facedata

def obtainSimilarityScore(img1,img2):
    detector = cv2.FeatureDetector_create("SIFT")
    descriptor = cv2.DescriptorExtractor_create("SIFT")
    skp = detector.detect(img1)
    skp, sd = descriptor.compute(img1, skp)
    tkp = detector.detect(img2)
    tkp, td = descriptor.compute(img2, tkp)
    num1 = 0
    for i in range(len(sd)):
          kp_value_min = np.inf
          kp_value_2min = np.inf
          for j in range(len(td)):
               kp_value = 0
               for k in range(128):
                     kp_value = (sd[i][k]-td[j][k]) *(sd[i][k]-td[j][k]) + kp_value
               if kp_value < kp_value_min:
                     kp_value_2min = kp_value_min
                     kp_value_min = kp_value
          if kp_value_min < 0.8*kp_value_2min:
               num1 = num1+1     
    num2 = 0
    for i in range(len(td)):
          kp_value_min = np.inf
          kp_value_2min = np.inf
          for j in range(len(sd)):
               kp_value = 0
               for k in range(128):
                     kp_value = (td[i][k]-sd[j][k]) *(td[i][k]-sd[j][k]) + kp_value
               if kp_value < kp_value_min:
                     kp_value_2min = kp_value_min
                     kp_value_min = kp_value
          if kp_value_min < 0.8*kp_value_2min:
               num2 = num2+1
    K1 = num1*1.0/len(skp)
    K2 = num2*1.0/len(tkp)
    SimilarityScore  = 100*(K1+K2)*1.0/2    
    return SimilarityScore

if __name__ =='__main__':
  filename_array = []  
  jpg_filenames = glob.glob('*.jpg')
  for filename in jpg_filenames:
      filename_array.append(filename)

  SSMatrix = [[100 for x in range(5)] for y in range(20)]
  for m in range(5):
    img1 = getFaceData(filename_array[5*m])
    for n in range(20):  
      img2 = getFaceData(filename_array[5+n])
      SSMatrix[n][m] = obtainSimilarityScore(img1,img2)
