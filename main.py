import cv2 as cv
import numpy as np
# import dlib

camera= cv.VideoCapture(0)
frame_width = int(camera.get(cv.CAP_PROP_FRAME_WIDTH))
frame_height = int(camera.get(cv.CAP_PROP_FRAME_HEIGHT))
fps = camera.get(cv.CAP_PROP_FPS) 


# def add_lipstick(img, landmarks, color=(255,0,0)):
#     img = img.copy()

#     lips_outer = np.array([landmarks[i] for i in range(48, 60)], dtype=np.int32)
#     lips_inner = np.array([landmarks[i] for i in range(60, 68)], dtype=np.int32)

#     mask = np.zeros_like(img)
#     cv.fillPoly(mask, [lips_outer], color)
#     cv.fillPoly(mask, [lips_inner], (0, 0, 0))

#     img = cv.GaussianBlur(img, (5, 5), 2)
#     img = cv.addWeighted(img, 0.9, mask, 0.2, 0)
#     return img
while True:
        ret, frame = camera.read() 

        if not ret:
            print("Error: Failed to read frame.")
            break
        fliped= cv.flip(frame, 1)
        img= cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        img= cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(img)
        gaus= cv.GaussianBlur(img, (15, 15), 0)
        t_lower = 40  
        t_upper = 90 
        canny = cv.Canny(gaus, t_lower, t_upper)
        Mask = cv.bitwise_not(canny)
        filmstrip= cv.imread("film.jpg")
        Mask= cv.cvtColor(Mask, cv.COLOR_GRAY2BGR)
        canny= cv.cvtColor(canny, cv.COLOR_GRAY2BGR)
        resized_mask = cv.resize(Mask, (404, 300))
        resized_canny=cv.resize(canny, (404, 300))
        resized_norm= cv.resize(frame, (404, 300))
        # resized_mask= cv.copyMakeBorder(resized_mask, 5, 5, 5, 5, cv.BORDER_CONSTANT, value=[40, 30, 29])
        # resized_canny= cv.copyMakeBorder(resized_canny, 5, 5, 5, 5, cv.BORDER_CONSTANT, value=[40, 30, 29])
        filmstrip[82:82+ resized_mask.shape[0], 165: 165+resized_mask.shape[1]]= resized_mask
        filmstrip[754:754+ resized_mask.shape[0], 165: 165+resized_mask.shape[1]]= resized_norm
        filmstrip[416:416+ resized_canny.shape[0], 165: 165+resized_canny.shape[1]]= resized_canny
        filmstrip[1090:1090+ resized_canny.shape[0], 165: 165+resized_canny.shape[1]]= resized_canny
        # cv.imshow('og', frame)
        cv.imshow("photobooth", filmstrip)
        # PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
        # face_detector = dlib.get_frontal_face_detector()
        # landmark_detector = dlib.shape_predictor(PREDICTOR_PATH)

        # gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # faces = face_detector(gray)

        # for face in faces:
        #     landmarks = landmark_detector(gray, face)
        #     points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)]
        # lips= add_lipstick(face, landmarks, color=(255, 0, 0))
        # cv.imshow('lips', lips)
        if cv.waitKey(50) & 0xFF == ord('q'):
            break
camera.release()
# out.release()
cv.destroyAllWindows()