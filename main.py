from ultralytics import YOLO

model = YOLO('yolov8n-cls.pt')  

model.train(data='C:/Users/user/logoDetection/Detection-Authenticity-of-logos-by-using-YOLOv8/data',
            epochs=13, imgsz=64)