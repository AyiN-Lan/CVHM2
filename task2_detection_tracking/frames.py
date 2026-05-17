import cv2
import os

cap=cv2.VideoCapture("result.mp4")
os.makedirs("frames_20s_to_26s",exist_ok=True)

fps=cap.get(cv2.CAP_PROP_FPS)
start_frame=int(16*fps)
end_frame=int(20*fps)

n=0
while True:
    ret,frame=cap.read()
    if not ret:break
    if n>=start_frame and n<=end_frame:
        cv2.imwrite(f"frames_20s_to_26s/{n:06d}.jpg",frame)
    if n>end_frame:break
    n+=1

cap.release()
