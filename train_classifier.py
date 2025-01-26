import numpy as np
from PIL import Image
import os
import cv2

def train_classifier(data_dir):
    path = [os.path.join(data_dir,f) for f in os.listdir(data_dir)]
    faces = []
    ids = []

    #convert image to grey scale
    for image in path:
        img = Image.open(image).convert('L')
        #convert image to array
        imageNp = np.array(img, 'uint8') 
        id = int(os.path.split(image)[1].split(".")[1])

        faces.append(imageNp)
        ids.append(id)

    ids = np.array(ids)

    # Train the classifier & save

    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.train(faces,ids)
    clf.write("classifier.xml")

# Call the function

train_classifier("data")