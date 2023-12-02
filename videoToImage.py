###pip install opencv-python
import cv2
import os
import random
def save_frame(video_path, frame_num, result_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return

    os.makedirs(os.path.dirname(result_path), exist_ok=True)

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)

    ret, frame = cap.read()

    if ret:
        cv2.imwrite(result_path, frame)
directory = './data/manipulated_sequences/DeepFakeDetection/raw'
count = 0
IMAGES_PER_VIDEO = 10
for filename in os.listdir(directory + '/videos'):
    if filename.endswith('.mp4'):
        for i in range(IMAGES_PER_VIDEO):
          path = os.path.join(directory + '/videos', filename)
          save_frame(path, random.randint(0,300), directory + f'/images/{count}.jpg')
          count += 1