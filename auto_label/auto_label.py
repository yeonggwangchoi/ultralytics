from ultralytics import YOLO
import cv2
import os

image_path = "/home/fc/glory_ws/ultralytics/auto_label/images"
label_path = "/home/fc/glory_ws/ultralytics/auto_label/labels"
model_path = "/home/fc/glory_ws/ultralytics/auto_label/models"
model = YOLO(os.path.join(model_path, 'best.pt'))

image_files = os.listdir(image_path)

for image_file in image_files:

    img = cv2.imread(image_file, cv2.IMREAD_COLOR)
    results = model(img)

    boxes = results[0].boxes

    for box in boxes:
        print("%d %.6f %.6f %.6f %.6f"%(box.cls, box.xywhn.to('cpu').numpy()[0][0],box.xywhn.to('cpu').numpy()[0][1],box.xywhn.to('cpu').numpy()[0][2],box.xywhn.to('cpu').numpy()[0][3]))


    f = open(os.path.join(label_path,image_file + ".txt"),"w")
    f.write("%d %.6f %.6f %.6f %.6f"%(box.cls, box.xywhn.to('cpu').numpy()[0][0],box.xywhn.to('cpu').numpy()[0][1],box.xywhn.to('cpu').numpy()[0][2],box.xywhn.to('cpu').numpy()[0][3]))
    f.close()