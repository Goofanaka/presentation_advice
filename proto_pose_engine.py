
import numpy as np
import cv2
import matplotlib.pyplot as plt
# Specify the paths for the 2 files
protoFile = 'D:/second_project/code_folders/openpose-master/models/pose/coco/pose_deploy_linevec.prototxt'
weightsFile = 'D:/second_project/code_folders/openpose-master/models/pose/coco/pose_iter_440000.caffemodel'

# Read the network into Memory
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

frame = cv2.imread('walking.jpeg')
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
print(frame.shape)
height, width = frame.shape[:2]

#pre-processing before put image into model
in_blob = cv2.dnn.blobFromImage(frame, 1.0/255, (width, height), (0,0,0), swapRB=False, crop=False)

net.setInput(in_blob)

#1D = Image_id, 2D = Confidence Maps and Part Affinity maps, 3D = height of output map, 4D = width
output = net.forward()

print(output.shape)

H_out = output.shape[2]
W_out = output.shape[3]

points = []
#확인용 코드
# ex_out1 = output[0][43]
# for i in range(10):
#     ex_out2 = output[0][i+1]
#     outs = cv2.bitwise_or(ex_out1, ex_out2)
#     ex_out1 = outs


for i in range(18):
    #confidence map of corresponding body's part
    probMap = output[0, i, :, :]

    #find global maximam of the probmap
    minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
    print(i ,'minVal : ', minVal, 'prob :', prob, 'minLoc : ', minLoc, 'point :', point)

    x = (width * point[0]) / W_out
    y = (height * point[1]) / H_out

    if prob > 0.1:
        cv2.circle(frame, (int(x), int(y)), 10, (255,255,255), thickness= -1, lineType=cv2.FILLED)
        cv2.putText(frame, f'{i}', (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2, lineType=cv2.LINE_AA)
        #add the point to the list if probability is greater than the threshold
        points.append((int(x), int(y)))
    else:
        points.append(None)

# plt.imshow(frame)
# plt.title('output_keypoints')
# plt.xticks([])
# plt.yticks([])
# plt.show()


frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
cv2.imshow('coco', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

#body points
BODY_PARTS = {}