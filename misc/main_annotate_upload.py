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
import torch 
print("CUDA is available:", torch.cuda.is_available())
app = FastAPI()

# model = YOLO("yolo11x-seg.pt")
model = YOLO("yolo11x.pt")
# Assuming `model` is your PyTorch model
model.to('cuda')

print('model on gpu' ,next(model.parameters()).is_cuda)

# all_hue_range = create_range_hue()

with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)

HOST_IP = config['HOST_IP'] 
PORT_NUM = config['PORT_NUM'] 
AVAIL_CLASS = config["AVAIL_CLASS"]
avail_class = AVAIL_CLASS.values()
COLOR_SERVER_PORT = config["COLOR_SERVER_PORT"]
import time 
from PIL import Image
import requests
current_dir = os.path.dirname(os.path.abspath(__file__))
SAVE_FOLDER = os.path.join(current_dir, 'static') 
os.makedirs(SAVE_FOLDER, exist_ok=True)

def fix_path(img_path):
     return img_path.replace('\\','/')


def color_recognize(image_path):
    print ("sending to server")
    url = f"http://{HOST_IP}:{COLOR_SERVER_PORT}/recognize-color"

    payload = {'image_path': image_path}
    files=[

    ]
    headers = {}

    response = requests.request("POST", url, headers=headers, data=payload, files=files)

    print(response.text)
    return response.json()["color"]

from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles

os.makedirs('static', exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")
for item in avail_class:
    print (item)
def padding_cv2_img(img):
    # Define the number of pixels to pad on each side
    top, bottom, left, right = 150, 150, 150, 150

    # Pad the image with a constant border (white color)
    padded_img = cv2.copyMakeBorder(
        img, 
        top, bottom, left, right, 
        borderType=cv2.BORDER_CONSTANT, 
        value=[255, 255, 255]  # white color in BGR format
    )
    return padded_img

@app.post("/detect-recognize-color") 
async def detect_recognize_color(target_class: str = Form(...), 
                                  file: UploadFile = File(...)):
    """
    given image have object:
    segment interested object with specific class 
    then find hue range of the object
    """
    # Create a file path to save the uploaded file
    UPLOAD_DIR = 'upload'
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    img_path = os.path.join(UPLOAD_DIR, str(uuid.uuid4()) + ".jpg")
    
    # Open the file asynchronously
    with open(img_path, 'wb') as out_file:
        # Read the contents of the uploaded file
        content = await file.read()  
        # Write the content to the file
        out_file.write(content)
    
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
            # padded_image = padding(img_path, a = 100) # PIL RGB result
            ori_img = cv2.imread(img_path)
            pad_img = padding_cv2_img(ori_img)
            batch_yolo_result = model.predict(conf=0.3, source=pad_img, save=False)
            for single_image_result in batch_yolo_result:
                # img = np.copy(single_image_result.orig_img)
                img = pad_img
                for ci, single_object_result in enumerate(single_image_result):
                    class_id = single_object_result.boxes.cls.tolist().pop()
                    label = single_object_result.names[class_id]
                    ########################################################
                    if label != target_class:
                        continue
                    conf = single_object_result.boxes.conf.tolist().pop()
                    # isolated = mask_img(img=img, c=single_object_result)    
                    # isolated = remove_padding(isolated)

                    # main_hue_range = find_main_color(isolated, all_hue_range) 
                    # rgb_class = RANGE_HUE_LABEL[main_hue_range]
                    x1, y1, x2, y2 = single_object_result.boxes.xyxy.tolist()[0]
                    cropped_image = img[int(y1):int(y2), int(x1):int(x2)]
                    temp_path = os.path.join(current_dir, "temp", str(uuid.uuid4()) + ".jpg")
                    os.makedirs("temp", exist_ok=True)
                    cv2.imwrite(temp_path, cropped_image)
                    rgb_class = color_recognize(temp_path)
                    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

                    # Draw the stick line (vertical line pointing down)
                    cv2.line(img, (int(x2), int(y2)), (int(x2), int(y2) + 40), (0, 0, 255), 2)

                    # Add label text at the top of the stick
                    cv2.putText(img, rgb_class, (int(x2), int(y2) + 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 0, 128), 2)                    ########################################################
                
                
                name_img = str(uuid.uuid4()) + ".jpg"
                res_path = os.path.join(SAVE_FOLDER, name_img) 
                
                cv2.imwrite(res_path, img)
                url = f"https://418e-2402-800-62d0-432e-ad25-d017-9455-ccb6.ngrok-free.app/static/{name_img}"
            return {
                        "status": 1,
                        "error_code": None,
                        "error_message": None,
                        "result": {
                            "res_path": url
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