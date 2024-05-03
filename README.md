# SUPRA

- This repository contains a pipeline for mapping 3D SUPRA videos to 2D map tracking, consisting of three main components:

  

  ## 1. Detection Model Fine-tuning (`./detection_model_finetuning`)

  This component is responsible for fine-tuning the YOLOv8 object detection model on the SUPRA videos. The fine-tuned model is used to detect and track objects in the 3D video frames.

  **Usage**:

  1. Place the labeled videos in json format and frame image files in the `data/` directory.
  2. Configure the training parameters in `config.py`.
  3. Run `train.py` to start the fine-tuning process.

  **Output**: A fine-tuned YOLOv8 model checkpoint.

  

  ## 2. Map Sort (`./Map_Sort`)

  This component takes the 3D video and the fine-tuned detection model as input, and maps the detected objects onto a 2D map. It also provides visualization for inspecting the mapped objects.

  **Usage**:

  1. Place your 3D video in the `data/` directory.
  2. Load the fine-tuned model checkpoint from the previous step.
  3. Run `track.py` to process the video and generate the 2D mapping.

  **Output**: A 2D mapping of the objects in the video, along with visualization tools.

  

  ## 3. Mapping Metrics (`./Mapsort_metrics`)

  This component calculates various metrics and statistics based on the 2D mapping generated in the previous step. These metrics can be used to analyze the spatial distribution, density, and other properties of the mapped objects.

  **Usage**:

  1. Load the 2D mapping output from the `Map_Sort` component.
  2. Run `compute_metrics.py` to calculate the desired metrics.

  **Output**: A report containing the computed mapping metrics.