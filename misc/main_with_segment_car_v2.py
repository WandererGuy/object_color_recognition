print("............. Initialization .............")
import ultralytics
ultralytics.checks()
from ultralytics import YOLO
import uuid
import os
import cv2
import numpy as np
from utils import create_range_hue, padding, mask_img, \
                    find_main_color, remove_padding, fix_path
from utils import RANGE_HUE_LABEL
import yaml
import uvicorn
from fastapi import FastAPI, HTTPException, Form, Request
app = FastAPI()
import torch 
print("CUDA is available:", torch.cuda.is_available())

model = YOLO("yolo11x-seg.pt")
# Assuming `model` is your PyTorch model
model.to('cuda')

print('model on gpu' ,next(model.parameters()).is_cuda)

os.makedirs('output', exist_ok=True)
all_hue_range = create_range_hue()

with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)

HOST_IP = config['HOST_IP'] 
PORT_NUM = 2001
AVAIL_CLASS = config["AVAIL_CLASS"]
avail_class = AVAIL_CLASS.values()
import time 
from PIL import Image

import shutil
if os.path.exists("demo"):
    shutil.rmtree("demo")            

@app.post("/segment-with-color-recognize-v2") 
async def segment_color_recognize_v2(target_class: str = Form(...), 
                                  img_path: str = Form(...)):
    """
    given image have object:
    segment interested object with specific class 
    then find hue range of the object
    """
    start = time.time()

    try:
            if target_class not in avail_class:
                return {
                        "status": 0,
                        "error_code": 400,
                        "error_message": f"target_class '{target_class}' unavailable.Available target class are {avail_class}",
                        "result": None
                        } 
            img_path = fix_path(img_path)
            final_res = {target_class: [0, None]}
            bgr_img = cv2.imread(img_path)
            if bgr_img is None:
                return {
                        "status": 0,
                        "error_code": 400,
                        "error_message": "Error: Image not loaded. Check the path",
                        "result": None
                        } 
            # uuid = str(uuid.uuid4())
            # out_folder = f'output/{uuid}'
            # os.makedirs(out_folder, exist_ok=True)
            # pad_img_path = os.path.join(out_folder, 'padded.png')
            # padding(img_path, pad_img_path)
            padded_image = padding(img_path, a = 100) # PIL RGB result
            
            import shutil 
            save_folder = "demo"
            os.makedirs(save_folder, exist_ok=True)

            batch_yolo_result = model.predict(conf=0.2, source=padded_image, save=False)
            for single_image_result in batch_yolo_result:
                img = np.copy(single_image_result.orig_img)
                for ci, single_object_result in enumerate(single_image_result):
                    class_id = single_object_result.boxes.cls.tolist().pop()
                    label = single_object_result.names[class_id]
                    ########################################################
                    if label != target_class:
                        continue
                    conf = single_object_result.boxes.conf.tolist().pop()
                    isolated = mask_img(img=img, c=single_object_result)    
                    isolated = remove_padding(isolated)

                    main_hue_range = find_main_color(isolated, all_hue_range) 
                    rgb_class = RANGE_HUE_LABEL[main_hue_range]
                    text_top = rgb_class
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    x1, y1, x2, y2 = single_object_result.boxes.xyxy.tolist()[0]
                    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

                    # Draw the stick line (vertical line pointing down)
                    cv2.line(img, (int(x2), int(y2)), (int(x2), int(y2) + 40), (0, 0, 255), 2)

                    # Add label text at the top of the stick
                    cv2.putText(img, text_top, (int(x2), int(y2) + 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (128, 0, 128), 2)

                    # Add confidence text at the top of the stick           
                    # # Define the font and scale
                    # font = cv2.FONT_HERSHEY_SIMPLEX
                    # font_scale = 0.5  # Smaller scale for smaller text
                    # thickness = 1  # Thickness of the text

                    # # Draw the rectangle
                    # cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

                    # # Get text size to adjust positioning
                    # (text_width, text_height), baseline = cv2.getTextSize(text_top, font, font_scale, thickness)

                    # # Calculate the position to center the text inside the rectangle
                    # text_x = int(x1 + (x2 - x1 - text_width) / 2)  # Horizontally centered
                    # text_y = int(y1 + (y2 - y1 + text_height) / 2)  # Vertically centered

                    # # Put the text inside the box
                    # cv2.putText(img, text_top, (text_x, text_y), font, font_scale, (0, 0, 255), thickness, cv2.LINE_AA)

                    ########################################################
                    
                
                name_img = os.path.basename(img_path).split(".")[0]
                cv2.imwrite(f"{save_folder}/{name_img}.jpg", img)

            return {
                    "status": 1,
                    "error_code": 200,
                    "error_message": None,
                    "result": final_res
                    }
    except Exception as e:

            return {
                    "status": 0,
                    "error_code": 500,
                    "error_message": e,
                    "result": None
                    }
def main():
    print('INITIALIZING FASTAPI SERVER')
    uvicorn.run(app, host=HOST_IP, port=int(PORT_NUM), reload=False)
    # uvicorn.run("sample_seg:app", host=host_ip, port=int(seg_port_num), reload=True)


if __name__ == "__main__":
    main()