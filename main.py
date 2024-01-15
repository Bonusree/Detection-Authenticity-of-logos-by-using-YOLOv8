from ultralytics import YOLO

model = YOLO('yolov8n-cls.pt')  # load a pretrained model (recommended for training)

model.train(data='C:/Users/user/logoDetection/Detection_Authenticity_of_logos_using_yolov8/Detection-Authenticity-of-logos-by-using-YOLOv8/data',
            epochs=13, imgsz=64)