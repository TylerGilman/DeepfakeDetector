###pip install opencv-python
import cv2
import os
import random

# Saves the frame_numth frame in the video at video_path to result_path
def save_frame(video_path, frame_num, result_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return

    os.makedirs(os.path.dirname(result_path), exist_ok=True)

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)

    ret, frame = cap.read()

    if ret:
        cv2.imwrite(result_path, frame)

# The path where thd videos to convert are stored
directory = './data/manipulated_sequences/DeepFakeDetection/raw'

count = 0
# The number of frames to extract from each video
IMAGES_PER_VIDEO = 10
# For each file in the directory
for filename in os.listdir(directory + '/videos'):
    # If that file is a video
    if filename.endswith('.mp4'):
        #Save IMAGES_PER_VIDEO random frames from it
        for i in range(IMAGES_PER_VIDEO):
          path = os.path.join(directory + '/videos', filename)
          save_frame(path, random.randint(0,300), directory + f'/images/{count}.jpg')
          count += 1