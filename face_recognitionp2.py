import cv2,sys,numpy,os
hi="haarcascade_frontalface_default.xml"
data="datasets"
print('Recognizing Face Please Be in sufficient Lights...')
(images,labels,names,id)=([],[],{},0)
for (subdir,dir,files) in os.walk(data):
    for subdir in dir:
        names[id]=subdir
        path=os.path.join(data,subdir)
        for filename in os.listdir(path):
            filepath=path+'/'+filename
            label=id
            images.append(cv2.imread(filepath,0))
            labels.append(label)
        id+=1
(width,height)=(130,100)
(images,labels)=[numpy.array(lis) for lis in[images,labels]]
model=cv2.LBPHFaceRecognizer_create()
model.train(images,labels)
face_cascade =cv2.CascadeClassifier(hi)
webcam=cv2.VideoCapture(0)
while True:
    (_,im)=webcam.read() 
    gray =cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces= face_cascade.detectMultiScale(gray,1.3,5)
    for(x,y,w,h) in faces:
        face=gray[y:y + h,x:x + w]
        face_resize= cv2.resize(face,(width,height))
        prediction=model.predict(face_resize)
    #prediction[1] is a confidence score
    #if the score is lower,it's better
    if prediction[1] <500:
        cv2.putText(im,'% s -%.0f'%(names[prediction[0]],prediction[1]),(x-10,y-10),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0))
    else:
        cv2.putText(im,'this person is not recognised',(x-10,y-10),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0))
    cv2.imshow("fmkjfk,",im)
    key =cv2.waitKey(10)
    if key ==27:
        break

