import cv2
import numpy as np

# Load the ArUco dictionary and parameters
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters()

# Marker size in meters (adjust based on your marker size)
marker_length = 0.067

# Object points in real world space for a single marker
objp = np.array([[0, 0, 0], [marker_length, 0, 0], [marker_length, marker_length, 0], [0, marker_length, 0]], dtype=np.float32)

# Video file path
video_path = 'video.mkv'

# Open the video file
cap = cv2.VideoCapture(video_path)

# Lists to store object points and image points for all frames
objpoints = []  # 3D points in real world space
imgpoints = []  # 2D points in image plane

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect markers
    corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    if ids is not None:
        for i in range(len(ids)):
            imgpoints.append(corners[i])
            objpoints.append(objp)

    # Display the frame
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Convert lists to numpy arrays
objpoints = np.array(objpoints, dtype=np.float32)
imgpoints = np.array(imgpoints, dtype=np.float32)

# Check if we have enough points for calibration
if len(imgpoints) >= 10:  # Ensure at least 10 points
    # Calibrate the camera
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # Save the calibration data
    np.save('camera_matrix.npy', camera_matrix)
    np.save('dist_coeffs.npy', dist_coeffs)

    print("Camera Matrix:\n", camera_matrix)
    print("Distortion Coefficients:\n", dist_coeffs)
else:
    print("Not enough points for calibration. Please capture more frames with ArUco markers.")
