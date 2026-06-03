import cv2 as cv
import numpy as np

camera= cv.VideoCapture(0)
frame_width = int(camera.get(cv.CAP_PROP_FRAME_WIDTH))
frame_height = int(camera.get(cv.CAP_PROP_FRAME_HEIGHT))
fps = camera.get(cv.CAP_PROP_FPS) 

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
        filmstrip[82:82+ resized_mask.shape[0], 165: 165+resized_mask.shape[1]]= resized_mask
        filmstrip[416:416+ resized_canny.shape[0], 165: 165+resized_canny.shape[1]]= resized_canny
        cv.imshow('og', frame)
        cv.imshow("photobooth", filmstrip)
        if cv.waitKey(50) & 0xFF == ord('q'):
            break
camera.release()
# out.release()
cv.destroyAllWindows()