from ultralytics import YOLO

model = YOLO('yolov8m.pt')

results = model.train(data='./data_custom.yaml', epochs=100, imgsz=640, batch=8)