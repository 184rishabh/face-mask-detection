from keras.models import load_model
import cv2
import numpy as np
model = load_model('face_model')

face_clsfr=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

source=cv2.VideoCapture(0)

labels_dict={0:'MASK',1:'NO MASK'}
color_dict={0:(0,255,0),1:(0,0,255)}
while(True):

    ret,img=source.read()
    faces=face_clsfr.detectMultiScale(img,1.3,5)  

    for (x,y,w,h) in faces:
        face_img=img[y:y+w,x:x+w]
        resized=cv2.resize(face_img,(64,64))
        normalized=resized/255.0
        reshaped=np.reshape(normalized,(1,64,64,3))
        result=model.predict(reshaped)
        if result>0.5:
           result=0;
        else:
            result=1;
        print(result)    
        cv2.rectangle(img,(x,y),(x+w,y+h),color_dict[result],2)
        cv2.rectangle(img,(x,y-40),(x+w,y),color_dict[result],-1)
        cv2.putText(img, labels_dict[result], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        
        
    cv2.imshow('LIVE',img)
    key=cv2.waitKey(1)
    
    if(key==27):
        break
        
cv2.destroyAllWindows()
source.release()
