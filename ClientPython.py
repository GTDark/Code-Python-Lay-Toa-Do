import socket
import sys
import time
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import argparse
import imutils
from cv2 import cv2
import matplotlib.pyplot as plt
import math
import numpy as np
import json

HOST = '192.168.0.115'  
PORT = 8000        

#Create  function to count frame
count=0
#Create 2 arrays to contain x and y
Setx=[]
Sety=[]
# Function: Find the midpoint between two points
def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

# Function: Find cos(phi)
def pos(a, b, c):
    cosphi = (a**2 + c**2 - b**2) / (2*a*c)
    y = a * float(cosphi)
    x = a * float(math.sqrt(abs(1-cosphi*cosphi)))
    return (x, y)
#Take link of ip webcam
url = "http://192.168.0.100:8080/"
cap = cv2.VideoCapture(url+"/video")
width = 150  # Define the width of the picture frame. Unit: cm

# Define names of each possible ArUco tag OpenCV supports
ARUCO_DICT = {
	"DICT_4X4_50": cv2.aruco.DICT_4X4_50,
	"DICT_4X4_100": cv2.aruco.DICT_4X4_100,
	"DICT_4X4_250": cv2.aruco.DICT_4X4_250,
	"DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
	"DICT_5X5_50": cv2.aruco.DICT_5X5_50,
	"DICT_5X5_100": cv2.aruco.DICT_5X5_100,
	"DICT_5X5_250": cv2.aruco.DICT_5X5_250,
	"DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
	"DICT_6X6_50": cv2.aruco.DICT_6X6_50,
	"DICT_6X6_100": cv2.aruco.DICT_6X6_100,
	"DICT_6X6_250": cv2.aruco.DICT_6X6_250,
	"DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
	"DICT_7X7_50": cv2.aruco.DICT_7X7_50,
	"DICT_7X7_100": cv2.aruco.DICT_7X7_100,
	"DICT_7X7_250": cv2.aruco.DICT_7X7_250,
	"DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
	"DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
	"DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
	"DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
	"DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
	"DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}

# Load the input image
start = time.time()
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("frame read failed")
        break


    orig = frame.copy() # Make a copy for further use
    size = (orig.shape)
    #print(size)
    point2 = (0, 0)
    point1 = (0, size[0])
    cv2.line(orig, point1, point2, (0, 255, 255), 2)

    #Find ratio between calculating and reality
    ratio = (dist.euclidean(point1, point2)) / width
    i = 0

    # Load the ArUCo dictionary, grab the ArUCo parameters, and detect the markers

    arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT["DICT_5X5_50"])
    arucoParams = cv2.aruco.DetectorParameters_create()
    (corners, ids, rejected) = cv2.aruco.detectMarkers(orig, arucoDict,
	    parameters=arucoParams)

    # Verify the ArUco marker was detected
    if len(corners) > 0:
	    # Flatten the ArUco IDs list
	    ids = ids.flatten()
	    # Loop over the detected ArUCo corners
	    for (markerCorner, markerID) in zip(corners, ids):
		    # Extract the marker corners (which are always returned in top-left, top-right, bottom-right, and bottom-left order)
		    corners = markerCorner.reshape((4, 2))
		    (topLeft, topRight, bottomRight, bottomLeft) = corners
		    # Convert each of the (x, y)-coordinate pairs to integers
		    topRight = (int(topRight[0]), int(topRight[1]))
		    bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
		    bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
		    topLeft = (int(topLeft[0]), int(topLeft[1]))

            # Draw the bounding box of the ArUCo detection
		    cv2.line(orig, topLeft, topRight, (0, 255, 0), 2)
		    cv2.line(orig, topRight, bottomRight, (0, 255, 0), 2)
		    cv2.line(orig, bottomRight, bottomLeft, (0, 255, 0), 2)
		    cv2.line(orig, bottomLeft, topLeft, (0, 255, 0), 2)
		    # Compute and draw the center (x, y)-coordinates of the ArUco marker
		    cX = int((topLeft[0] + bottomRight[0]) / 2.0)
		    cY = int((topLeft[1] + bottomRight[1]) / 2.0)
		    cv2.circle(orig, (cX, cY), 4, (0, 0, 255), -1)
		    # Draw the ArUco marker ID on the image
		    cv2.putText(orig, str(markerID),
			    (topLeft[0], topLeft[1] - 15), cv2.FONT_HERSHEY_SIMPLEX,
			    0.5, (0, 255, 0), 2)
		    #print("[INFO] ArUco marker ID: {}".format(markerID))
    if len(corners) > 0:
        t1 = midpoint((topRight[0], topRight[1]), (topLeft[0], topLeft[1]))
        t2 = midpoint((bottomRight[0], bottomRight[1]), (bottomLeft[0], bottomLeft[1]))
        t3 = midpoint(t1, t2) # t3 is the center to the ArUCo marker

        cv2.circle(orig, (int(t3[0]), int(t3[1])), 5, (255, 0, 0), -1)

        cv2.line(orig, point1, (int(t3[0]), int(t3[1])), (0, 255, 255), 2)
        cv2.line(orig, point2, (int(t3[0]), int(t3[1])), (0, 255, 255), 2)

        D1 = (dist.euclidean(point1, (int(t3[0]), int(t3[1]))))/ratio
        D2 = (dist.euclidean(point2, (int(t3[0]), int(t3[1]))))/ratio

        (mX1, mY1) = midpoint(point1, (int(t3[0]), int(t3[1])))
        (mX2, mY2) = midpoint(point2, (int(t3[0]), int(t3[1])))

        # Print the distances from two corners of the Image to the center of the ArUCo marker
        cv2.putText(orig, "{:.1f}cm".format(D1), (int(mX1), int(mY1 - 40)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0), 2)
        cv2.putText(orig, "{:.1f}cm".format(D2), (int(mX2), int( mY2 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

        # Find the coordinates of the marker
        x = round(pos(D1, D2, width)[0], 1)
        y = round(pos(D1, D2, width)[1], 1)
        count+=1
        if (count%30==1):
            Setx.append(x)
            Sety.append(y)
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server_address = (HOST, PORT)
            print('connecting to %s port ' + str(server_address))
            s.connect(server_address)
            x1 = json.dumps(x)
            msg=' '
            y1 = json.dumps(y)            
            s.sendall(bytes(x1, "utf8"))
            s.sendall(bytes(msg, "utf8"))
            s.sendall(bytes(y1, "utf8"))  
            s.close()     
        # Print the coordinates to the Output Image
        cv2.putText(orig, "x={:.1f}cm".format(x), (int(t3[0]-100), int(t3[1]-80)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 2, 2), 2)
        cv2.putText(orig, "y={:.1f}cm".format(y), (int(t3[0]-100), int(t3[1]-60)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 2, 2), 2)

        # Print the size of the ArUCo marker to the screen
        Dqr = (dist.euclidean((topRight[0], topRight[1]), (topLeft[0], topLeft[1])))/ratio
        cv2.putText(orig, "{:.1f}cm".format(Dqr), (int(t1[0]+20), int(t1[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 2, 2), 2)
    cv2.imshow("Result",orig)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  


print(Setx)
print(Sety)
end = time.time()
print(end - start)
cap.release()
cv2.destroyAllWindows()

