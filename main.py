print("............. Initialization .............")
import cv2
from utils import create_range_hue, find_main_color
from utils import RANGE_HUE_LABEL
import yaml
import uvicorn
import time 
from fastapi import FastAPI, HTTPException, Form, Request
app = FastAPI()

all_hue_range = create_range_hue()

with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)

HOST_IP = config['HOST_IP'] 
PORT_NUM = config['PORT_NUM'] 

@app.post("/color-recognize") 
async def color_recognize(img_path_ls: str = Form(...)):
    '''
    find hue color of isolated segment object in image 
    '''
    new_img_path_ls = eval(img_path_ls)

    try:
        img_dict = {}
        start = time.time()
        for img_path in new_img_path_ls:
            img = cv2.imread(img_path)
            main_hue_range = find_main_color(img, all_hue_range) 
            color = RANGE_HUE_LABEL[main_hue_range]
            img_dict[img_path] = color
        print ('infer time: ',time.time() - start)

        return {
                    "status": 1,
                    "error_code": None,
                    "error_message": None,
                    "result": {
                        "images_color": img_dict
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