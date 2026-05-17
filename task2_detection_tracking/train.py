from ultralytics import YOLO
model=YOLO('yolov8s.pt')
model.train(data='VisDrone.yaml',epochs=30,imgsz=640,batch=8,device='cuda',workers=2,val=True)
