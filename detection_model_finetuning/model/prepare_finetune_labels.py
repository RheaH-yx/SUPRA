import json
import cv2
import os
import random

# Define the class mapping
class_mapping = {
    "Person": 0,
    "Person0": 1,
    "Person1": 2,
    "Person2": 3,
    "Person3": 4,
    "Person4": 5
}


os.makedirs('./train/images', exist_ok=True)
os.makedirs('./val/images', exist_ok=True)
os.makedirs('./train/labels', exist_ok=True)
os.makedirs('./val/labels', exist_ok=True)


original_width = 1920
original_height = 1080
label_width = 100
label_height = 100

scale_x = original_width / label_width
scale_y = original_height / label_height


# Load the JSON file (labeled output)
with open('only_person_camF.json') as f:
    data = json.load(f)

# Open the video file
video_path = "./SH_R2_CamF.mp4"
cap = cv2.VideoCapture(video_path)

print('Start constructing training and testing datasets')
frame_labels = {}
for item in data:
    for box in item['box']:
        for seq in box['sequence']:
            frame_time = seq['time'] * 1000  # Convert time to milliseconds
            frame_num = seq['frame']
            x = seq['x'] * scale_x  # Convert x-coordinate to original resolution
            y = seq['y'] * scale_y  # Convert y-coordinate to original resolution
            width = seq['width'] * scale_x  # Convert width to original resolution
            height = seq['height'] * scale_y  # Convert height to original resolution
            label = box['labels'][0]

            if frame_num not in frame_labels:
                frame_labels[frame_num] = []

            # Set video position to frame_time in milliseconds
            cap.set(cv2.CAP_PROP_POS_MSEC, frame_time)
            ret, frame = cap.read()

            if ret:
                img_height, img_width, _ = frame.shape
                x_center = (x + width / 2) / img_width
                y_center = (y + height / 2) / img_height
                width_norm = width / img_width
                height_norm = height / img_height
                class_id = class_mapping[label]
                frame_labels[frame_num].append(f'{class_id} {x_center} {y_center} {width_norm} {height_norm}')

                # Assume first half of the frames are for training and the second half for validation
                if frame_num <= 4000:
                    img_path = f'./train/images/frame_{frame_num:04d}.jpg'
                else:
                    img_path = f'./val/images/frame_{frame_num:04d}.jpg'

                cv2.imwrite(img_path, frame)

cap.release()

# Create label files for all frames
for frame_num in frame_labels:
    if frame_num <= 4000:
        label_path = f'./train/labels/frame_{frame_num:04d}.txt'
    else:
        label_path = f'./val/labels/frame_{frame_num:04d}.txt'

    with open(label_path, 'w') as f:
        f.write('\n'.join(frame_labels[frame_num]) + '\n')

print('Completed!')