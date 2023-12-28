import face_recognition
import cv2 
import pickle
import os 

folderPath="Images"
pathList=os.listdir(folderPath)
print(pathList)
# for getting the complete path of image
imgList=[]
# for getting the id through image name 
studentIds=[]
for path in pathList:
    imgList.append(cv2.imread(os.path.join(folderPath,path)))
    # print(path)
    # print(os.path.splitext(path)[0])
    studentIds.append(os.path.splitext(path)[0])
# print(len(imageList))
print(studentIds)

def findEncodings(imageList):
    
    encodeList=[]
    for img in imageList:
        # opencv uses bgr and face_recognition uses rgb
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode=face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

print("Encoding Started...")
encodeListKnown=findEncodings(imgList)
# print(encodeListKnown)

encodeListKnownWithIds=[encodeListKnown,studentIds]
print("Encoding Completed")

# creating a file in which we will addd the encoded data of image and id of the student
file = open("EncodeFile.p","wb")
# sending the data to this file through pickle
pickle.dump(encodeListKnownWithIds,file)
file.close()
print("File Saved")
