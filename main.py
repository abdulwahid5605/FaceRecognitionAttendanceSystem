import cv2
import os
import pickle
import face_recognition
import numpy as np
import cvzone

# Open the camera
cap = cv2.VideoCapture(0)

# Set the width and height of the camera
cap.set(3, 640)  
cap.set(4, 720)   

# importing the mode images into a list
folderModePath="Resources/Modes"
modePathList= os.listdir(folderModePath)
imgModeList=[]
for path in modePathList:
    imgModeList.append(cv2.imread(os.path.join(folderModePath,path)))

# print(len(imgModeList))
# print(modePathList)

# studentIds and encoded data of images is saved in pickle
# Now loading that encoded file
print("Encode file loading")
file=open("EncodeFile.p","rb")
encodeListKnownWithIds=pickle.load(file)
file.close()
# fetching the encodings and student ids
encodeListKnown,studentIds=encodeListKnownWithIds
print(studentIds)
print("Encode file loaded")

    
# Storing bg
imgBackground = cv2.imread("Resources/background.png")

while True:
    # img variable is storing the camera
    success, img = cap.read()

    # now resizing the image that is captured by the camera because large images require too much computation power
    # resizing using the scale values(0.25 is the smallest)
    imgS=cv2.resize(img,(0,0),None,0.25,0.25)

    # now fetching the face location from the entire frame using face recognition
    faceCurFrame=face_recognition.face_locations(imgS)
    encodeCurFrame=face_recognition.face_encodings(imgS,faceCurFrame)


    # storing img at some specific points of the background
    imgBackground[175:175+480,94:94+640]=img

    imgBackground[38:38+633,833:833+414]=imgModeList[0]

    # for summing up two loops in a single loop
    for encoFace, faceLoc in zip(encodeCurFrame,faceCurFrame):
        # answer will be true or false. It will just tell face is matching or not
        matches=face_recognition.compare_faces(encodeListKnown,encoFace)
        # answer will be integer. It will just tell difference between the face stored and the current face
        faceDis=face_recognition.face_distance(encodeListKnown,encoFace)
        # print("matches",matches)
        # print("faceDis",faceDis)

        # Now finding the index where the image in the list becomes true and that is the least value of faceDist
        matchIndex=np.argmin(faceDis)
        # print(matchIndex)
        # print(studentIds[matchIndex])
        # rectangle thickness = rt
        y1,x2,y2,x1=faceLoc
        # multiplying by 4 because size is reduced
        y1,x2,y2,x1=y1*4,x2*4,y2*4,x1*4

        # applying padding
        bbox=0+x1,0+y2,x2-x1,y2-y1
        cvzone.cornerRect(imgBackground,bbox,rt=0)

        # with the help of matchIndex we will draw a rectangle around our face in live camera using cvzone
        if matches[matchIndex]:
            print("Known face detected")
    if not success:
        print("Error reading frame from camera")
        break

    # Showing main bg
    cv2.imshow("Face Attendance", imgBackground)

    # Exit the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
