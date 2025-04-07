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
PORT_NUM = config['PORT_NUM'] 
AVAIL_CLASS = config["AVAIL_CLASS"]
avail_class = AVAIL_CLASS.values()
import time 
@app.post("/segment-with-color-recognize") 
async def segment_color_recognize(target_class: str = Form(...), 
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

            batch_yolo_result = model.predict(conf=0.2, source=padded_image, save=False)
            for single_image_result in batch_yolo_result:
                img = np.copy(single_image_result.orig_img)
                for ci, single_object_result in enumerate(single_image_result):
                    class_id = single_object_result.boxes.cls.tolist().pop()
                    label = single_object_result.names[class_id]
                    conf = single_object_result.boxes.conf.tolist().pop()
                    isolated = mask_img(img=img, c=single_object_result)    
                    isolated = remove_padding(isolated)
                    
                    main_hue_range = find_main_color(isolated, all_hue_range) 
                    rgb_class = RANGE_HUE_LABEL[main_hue_range]
                    # save_path_isolated = os.path.join(out_folder, f'{label}_{ci}_{conf:.2f}.png')
                    # cv2.imwrite(save_path_isolated, isolated) # isolated is BGR, yolo output is BGR
                    # print ('save_path_isolated', save_path_isolated)
                    # print ('main_hue_range', rgb_class)
                    if label == target_class:
                        if conf > final_res[target_class][0]:
                            final_res[target_class][0] = conf
                            final_res[target_class][1] = rgb_class
            if final_res[target_class][0] == 0:
                return {
                        "status": 0,
                        "error_code": 400,
                        "error_message": "cannot detect target_class object to segment. Check if correct image have target object in it",
                        "result": None
                        }     
            color = final_res[target_class][1]
            print ('infer time ', time.time() - start)

            return {
                        "status": 1,
                        "error_code": None,
                        "error_message": None,
                        "result": {
                            "hue_range": color
                        }
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