import cv2 as cv
import numpy as np
import dlib



PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
face_detector = dlib.get_frontal_face_detector()
try:
    landmark_detector = dlib.shape_predictor(PREDICTOR_PATH)
except RuntimeError:
    print(f"Error: Clear path required. Ensure '{PREDICTOR_PATH}' is in the same directory.")
    # exit()

camera= cv.VideoCapture(0)
frame_width = int(camera.get(cv.CAP_PROP_FRAME_WIDTH))
frame_height = int(camera.get(cv.CAP_PROP_FRAME_HEIGHT))
fps = camera.get(cv.CAP_PROP_FPS) 


def add_lipstick(img, landmarks, color):
    img = img.copy()

    lips_outer = np.array([landmarks[i] for i in range(48, 60)], dtype=np.int32)
    lips_inner = np.array([landmarks[i] for i in range(60, 68)], dtype=np.int32)

    mask = np.zeros_like(img)
    cv.fillPoly(mask, [lips_outer], color)
    cv.fillPoly(mask, [lips_inner], (0, 0, 0))

    img = cv.GaussianBlur(img, (5, 5), 2)
    img = cv.addWeighted(img, 0.9, mask, 0.2, 0)
    return img

while True:
        ret, frame = camera.read() 

        if not ret:
            print("Error: Failed to read frame.")
            break
        
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = face_detector(gray)
        points= []
        for face in faces:
            landmarks = landmark_detector(gray, face)
            points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)]
            break
        if len(points) == 68:
        # Generate Lipstick Frame
            lipstick = add_lipstick(frame, points, (0, 255, 0))
        else:
        # Fallback if no face is detected so the app doesn't crash
            lipstick = frame.copy()
            cv.putText(lipstick, "No Face Detected", (30, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2) 
        img= cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        img= cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(img)
        gaus= cv.GaussianBlur(img, (15, 15), 0)
        t_lower = 40  
        t_upper = 90 
        canny = cv.Canny(gaus, t_lower, t_upper)
        Mask = cv.bitwise_not(canny)
        # lipstick= add_lipstick(frame, points, (0, 255, 0))
        filmstrip= cv.imread("film-4.jpg")
        Mask= cv.cvtColor(Mask, cv.COLOR_GRAY2BGR)
        canny= cv.cvtColor(canny, cv.COLOR_GRAY2BGR)
        resized_mask = cv.resize(Mask, (542, 300))
        resized_canny=cv.resize(canny, (542, 300))
        resized_norm= cv.resize(frame, (542, 300))
        resized_lips= cv.resize(lipstick, (542, 300))
        # resized_mask= cv.copyMakeBorder(resized_mask, 5, 5, 5, 5, cv.BORDER_CONSTANT, value=[40, 30, 29])
        # resized_canny= cv.copyMakeBorder(resized_canny, 5, 5, 5, 5, cv.BORDER_CONSTANT, value=[40, 30, 29])
        filmstrip[81:81+ resized_mask.shape[0], 227: 227+resized_mask.shape[1]]= resized_lips
        filmstrip[755:755+ resized_mask.shape[0], 227: 227+resized_mask.shape[1]]= resized_norm
        filmstrip[415:415+ resized_canny.shape[0], 227: 227+resized_canny.shape[1]]= resized_canny
        filmstrip[1091:1091+ resized_canny.shape[0], 227: 227+resized_canny.shape[1]]= resized_mask
        # cv.imshow('og', frame)
        cv.imshow("photobooth", filmstrip)
        
        if cv.waitKey(50) & 0xFF == ord('q'):
            break
camera.release()
# out.release()
cv.destroyAllWindows()