# Generate data set
import cv2
def general_dataset():
    face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    def face_cropped(img): #crop and convert to grey scale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray,1.3,5) 
       
        # scaling factor = 1.3
        #Minumum neighbour = 5

        if faces is ():
            return None
        for(x,y,w,h) in faces:
            cropped_face=img[y:y+h, x:x+w]
        return cropped_face
    
    cap = cv2.VideoCapture(0) #open webcam
    id=1 #id of first authorised person (change it to 2 if required 2 persons)
    img_id=0 #number of image of each authorized person

    # if there are any image increase id value by 1
    while True:
        ret, frame=cap.read()
        if face_cropped(frame) is not None:
            img_id+=1
            face=cv2.resize(face_cropped(frame),(200,200))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            file_name_path="data/user."+str(id)+"."+str(img_id)+".jpeg"
            cv2.imwrite(file_name_path, face)
            cv2.putText(face, str(img_id), (50,50), cv2.FONT_HERSHEY_COMPLEX,1, (0,255,0),2)
            # (50,50) is the origin point from where text is to written
            # font scale = 1
            # thickness = 2

            cv2.imshow("Cropped face", face)
            if cv2.waitKey(1)==27 or int(img_id)==400: # 27 = Esc key for loop break & 200 = number of images to be taken
                break
    cap.release()
    cv2.destroyAllWindows()
    print("collecting samples is completed......")
#general_dataset() 