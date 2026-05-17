import cv2
from ultralytics import YOLO
import logging
logging.basicConfig(level=logging.INFO,format='%(levelname)s - %(message)s',handlers=[logging.FileHandler('track_count.log'),logging.StreamHandler()])
CLASS_NAMES={0:"pedestrian",1:"people",2:"bicycle",3:"car",4:"van",5:"truck",6:"tricycle",7:"awning-tricycle",8:"bus",9:"motor"}
DETECT_CONF=0.3
DETECT_IOU=0.5
MIN_BOX_AREA=1000
def main():
    model=YOLO("./runs/detect/train/weights/best.pt")
    cap=cv2.VideoCapture("./test_video.mp4")
    if not cap.isOpened():
        logging.error("Cannot open video")
        return
    fps=int(cap.get(cv2.CAP_PROP_FPS))
    w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out=cv2.VideoWriter("result.mp4",cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h))
    line_x=w*5//8
    crossed_ids=dict()
    counts=dict()
    id_state=dict()
    frame_count=0
    while cap.isOpened():
        ret,frame=cap.read()
        if not ret:
            break
        frame_count+=1
        if frame_count%10==0:
            logging.info(f"Processing frame {frame_count}/{total_frames}")
        results=model.track(frame,persist=True,conf=DETECT_CONF,iou=DETECT_IOU,tracker="bytetrack.yaml",verbose=False)
        if results[0].boxes.id is not None:
            boxes=results[0].boxes.xyxy.cpu().numpy()
            track_ids=results[0].boxes.id.int().cpu().numpy()
            cls_list=results[0].boxes.cls.cpu().numpy()
            for box,track_id,c in zip(boxes,track_ids,cls_list):
                c=int(c)
                if c not in crossed_ids:
                    crossed_ids[c]=set()
                    counts[c]=0
                x1,y1,x2,y2=box.astype(int)
                box_area=(x2-x1)*(y2-y1)
                if box_area<MIN_BOX_AREA:
                    continue
                is_crossing=(x1<line_x<x2)
                class_name=CLASS_NAMES.get(c,f"Class{c}")
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                label_text=f"{class_name} ID:{track_id}"
                cv2.putText(frame,label_text,(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
                if track_id not in id_state:
                    id_state[track_id]=is_crossing
                else:
                    if id_state[track_id]!=is_crossing:
                        if track_id not in crossed_ids[c]:
                            crossed_ids[c].add(track_id)
                            counts[c]+=1
                            logging.info(f"Class {class_name} (ID {track_id}) crossed line, current count: {counts[c]}")
                    id_state[track_id]=is_crossing
        total_count=sum(counts.values())
        cv2.line(frame,(line_x,0),(line_x,h),(0,0,255),2)
        cv2.putText(frame,f"Total Count: {total_count}",(50,80),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),3)
        out.write(frame)
    cap.release()
    out.release()
    logging.info("Processing completed")
    logging.info("Category counts:")
    for cls_id,cnt in sorted(counts.items()):
        class_name=CLASS_NAMES.get(cls_id,f"Class{cls_id}")
        logging.info(f"{class_name}: {cnt}")
    logging.info(f"Total count: {sum(counts.values())}")
if __name__=="__main__":
    main()
