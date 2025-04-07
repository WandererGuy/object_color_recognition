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

# model = YOLO("yolo11x-seg.pt")
model = YOLO("yolo11x.pt")
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
from PIL import Image
import requests

def color_recognize(image_path):

    url = f"http://{HOST_IP}:4003/recognize-color"

    payload = {'image_path': image_path}
    files=[

    ]
    headers = {}

    response = requests.request("POST", url, headers=headers, data=payload, files=files)

    print(response.text)
    return response.json()["color"]

current_dir = os.path.dirname(os.path.abspath(__file__))


from fastapi import FastAPI, UploadFile, File
SAVE_FOLDER = os.path.join(current_dir, 'static') 
os.makedirs(SAVE_FOLDER, exist_ok=True)

from fastapi.staticfiles import StaticFiles

os.makedirs('static', exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.post("/detect-recognize-color") 
async def detect_recognize_color(
    #  target_class: str = Form(...), 
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
    """
    given image have object:
    segment interested object with specific class 
    then find hue range of the object
    """
    start = time.time()
    try:
            # if target_class not in avail_class:
            #     return {
            #             "status": 0,
            #             "error_code": 400,
            #             "error_message": f"target_class '{target_class}' unavailable.Available target class are {avail_class}",
            #             "result": None
            #             } 
            # final_res = {target_class: [0, None]}

            img_path = fix_path(img_path)
            bgr_img = cv2.imread(img_path)
            if bgr_img is None:
                return {
                        "status": 0,
                        "error_code": 400,
                        "error_message": "Error: Image not loaded. Check the path",
                        "result": None
                        } 
            padded_image = padding(img_path, a = 100) # PIL RGB result

            batch_yolo_result = model.predict(conf=0.2, source=padded_image, save=False)
            for single_image_result in batch_yolo_result:
                img = np.copy(single_image_result.orig_img)
                for ci, single_object_result in enumerate(single_image_result):
                    
                    class_id = single_object_result.boxes.cls.tolist().pop()
                    label = single_object_result.names[class_id]
                    
                    ########################################################
                    if label not in ["bicycle","car","motorcycle","bus","truck"]:
                        continue
                    conf = single_object_result.boxes.conf.tolist().pop()
                    x1, y1, x2, y2 = single_object_result.boxes.xyxy.tolist()[0]
                    cropped_image = img[int(y1):int(y2), int(x1):int(x2)]
                    temp_path = os.path.join(current_dir, "temp", str(uuid.uuid4()) + ".jpg")
                    os.makedirs("temp", exist_ok=True)
                    cv2.imwrite(temp_path, cropped_image)
                    print ("sending request to color recognize server")
                    rgb_class = color_recognize(temp_path)

                    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

                    # Draw the stick line (vertical line pointing down)
                    cv2.line(img, (int(x2), int(y2)), (int(x2), int(y2) + 40), (0, 0, 255), 2)

                    # Add label text at the top of the stick
                    cv2.putText(img, rgb_class, (int(x2), int(y2) + 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 0, 128), 2)
                
                name_img = str(uuid.uuid4()) + ".jpg"
                res_path = os.path.join(SAVE_FOLDER, name_img) 
                
                cv2.imwrite(res_path, img)

            print ('infer time ', time.time() - start)

            url = f"http://{HOST_IP}:{PORT_NUM}/static/{name_img}"
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