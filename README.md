# SUPRA

- This repository contains a pipeline for mapping 3D SUPRA videos to 2D map tracking, consisting of three main components:

  

  ## 1. Object Detection (`./detection_model_finetuning`)

  This component is responsible for fine-tuning the YOLOv8 object detection model on the SUPRA videos. The fine-tuned model is used to detect and track objects in the 3D video frames.

  The yolov8m.pt is already fine-tuned on a video and is ready for tracking. You can skip the following **Model Fine-tuning Usage** if you don't intend to modify the current model.

  #### **Model Fine-tuning Usage**:

  1. <u>**Prepare input:**</u> Place the labeled video output in json format  (i.e. /detection_model_finetuning/model/only_person_camF.json) and original video (i.e. /detection_model_finetuning/model/SH_R2_CamF.mp4 ) under /detection_model_finetuning/model

  2. **<u>Construct training and testing sets:</u>** go to prepare_finetune_labels.py. YOLOv8 is finetuned on images, so we will need to extract frames of the video and the bounding box coordinates of target objects in that frame image. The purpose of this python script is to construct an 'images' and a "labels" folder for each of the training and testing set. You may need to modify these lines for the new labels:

     - line 22-26: the original width and height are the resolution of the original video. If your labeling output is already normalized (in my case, I use label studio and they normalize each frame to 100x100), you'll need to convert it back to the original resolution
     - line 32-34: change labelling output in json format to the new path
     - line 36-37: change video path

     You can now run this script `python3 prepare_finetune_labels.py` and it will store frame images to the images folder and labeling coordinates to the labels folder. 

     The code separates training and testing frames manually by selecting the first 4000 frames for training. You can rewrite some code (lines 80-87) to make the separation more random for better training performance.

  3. **<u>Fine-tune:</u>** Go to `data_custom.yaml` and change the absolute path of model directory. Depending on your object class of interested, you may need to change the number of classes (line 5) and names (line 6). After that, run `fine_tuning.py`.

  **Output**: A fine-tuned YOLOv8 model checkpoint.

  

  #### Object Detection and Tracking Usage:

  1. For <u>object detection</u> only: Open a command window and run: `yolo task=detect mode=predict model=yolov8m.pt show=True conf=0.5 source=SH_R2_CamF.mp4 save_txt=True`

     **Output**: A video with predicted labels and txt files with coordinates for the bounding boxes for each frame can be found under ./runs/detect/labels

  2. For <u>object tracking</u>: 

     in command window run: `yolo track model=path/to/best.pt source="path/to/video.mp4" tracker="bytetrack.yaml" save_txt=True`

     **Output**: A video with predicted labels and txt files with coordinates for the bounding boxes for each frame can be found under ./runs/track/labels

     

  > [!NOTE]
  >
  > The output from YOLOv8 model is multiple txt files for each frame. However, the mapping in the next part only takes in a single txt file. To combine all the txt files from YOLOv8 model, run aggregate_txt.py, you may need to change the directory to the YOLOv8 tracking result txt files (line 4) and the resolution of the original video (line 6-7)

  

  ## 2. Mapping Visualization (`./Map_Sort`/examples)

  This component takes the 3D video and the fine-tuned detection model as input, and maps the detected objects onto a 2D map. It also provides visualization for inspecting the mapped objects.

  **Usage**:

  1. Place the aggregated txt file from last step in the `data/SUPRA` directory.

  2. Go to track.py, which generates a video of the 2d mapping. Before running this script, you may need to modify/create a few files defined from line 25 to 31

     - `video_path`: This is the path to the original 3D video file.

     - `map_path`: This is the path to a 2D map sketch of the room.

     - `dets_path`: This is the path to the combined 3D tracking results from YOLOv8, obtained in the previous step.

     - `point_map_path`: You can randomly select four or more points in the video and find their corresponding coordinates on the 2D map. The text file should have the following format:

       ```
       #frame_x,frame_y,map_x,map_y
       187,476,80,48
       1054,385,477,48
       266,1096,80,292
       1330,830,477,292
       ```

       Where `frame_x` and `frame_y` are the coordinates in the 3D video, and `map_x` and `map_y` are the coordinates on the 2D map. Each line is a coordinate in the video screenshot and its corresponding coordinate in the 2d map. You can use a painting tool to retrieve these coordinates.

     - `entry_polys_path`: This is the path to a file containing the coordinates of a bounding box for the door or entry area where people might enter. This is used to ensure that newly detected objects appear from this entry point.

     - `tracking_output_path`: This is the path where you want to store the 2D mapped coordinates.

     - `video_output_path`: This is the path where you want to store the output video file containing the results.

  3. Now you can run track.py to get a video of the 2d map. 

     > [!NOTE]
     >
     > This script assumes that there are only 5 people in the room (one enemy and four soldiers). If that is not the case for the new video, you will need to change line 81 `dets = [Detection(det[2:6], det[6]) for det in dets if int(det[1]) in [1, 2, 3, 4, 5]]` which checks if class id (det[1], the second element of each line in aggregated_results.txt) belongs to any of [1, 2, 3, 4, 5]. If there are more soldiers, you probably need to add 6, 7, etc. If you are not interested in the mapping of enemy (for example, in the next step Metric Calculation, we don’t want to include the enemy when measuring soldier performance), you can remove 1. 
     >
     > The meaning of each class number depends on how you label and train your model. The current model assumes that class id 1 is the enemy, class id 2 is the soldier with purple helmet, class id 3 is the soldier with pink helmet,  class id 4 is the soldier with green helmet, class id 5 is the soldier with blue helmet.

     

  ## 3. Mapping Metrics (`./Mapsort_metrics`)

  This component takes the 3D video and the fine-tuned detection model as input, and maps the detected objects onto a 2D map. It also calculates various metrics and statistics based on the 2D mapping. If you would like to only calculate the metrics without visualization, you can skip the previous step (2. Mapping Visualization)

  **Usage**:

  1. Metric calculation for a single video: track.py

     same as 2. Mapping Visualization –> Usage 2, modify line 23-31. boundary_path is the bounding box coordinates for the room. pod_path is a list of ideal points for soldiers to stop at (If you don’t know this now, just put in some dummy values)

     **Output**: A report containing the computed mapping metrics.

  2. Metric calculation for multiple videos: track_multiple.py

     useful for calculating metrics for a whole squad

  > [!NOTE]
  >
  > Similar as the second step – Mapping Visualization, change the person of interest for performance measurement at line 66. 

  