import ultralytics
ultralytics.checks()
from ultralytics import YOLO
import uuid
import os
import cv2
import numpy as np
from utils import create_range_hue, padding, mask_img, find_main_color
from utils import RANGE_HUE_LABEL
model = YOLO("yolo11x-seg.pt")
os.makedirs('output', exist_ok=True)
all_hue_range = create_range_hue()



if __name__ == '__main__':
    target_class = "bear"
    final_res = {target_class: [0, None]}
    img_path = '/home/ai-ubuntu/hddnew/Manh/obj_color/images/banana.jpg'
    bgr_img = cv2.imread(img_path)
    if bgr_img is None:
        print("Error: Image not loaded. Check the path.")
    # uuid = str(uuid.uuid4())
    # out_folder = f'output/{uuid}'
    # os.makedirs(out_folder, exist_ok=True)
    # pad_img_path = os.path.join(out_folder, 'padded.png')
    # padding(img_path, pad_img_path)
    padded_image = padding(img_path) # PIL RGB result

    results = model.predict(conf=0.2, source=padded_image, save=False)
    for r in results:
        img = np.copy(r.orig_img)
        for ci, c in enumerate(r):
            label = c.names[c.boxes.cls.tolist().pop()]
            conf = c.boxes.conf.tolist().pop()
            isolated = mask_img(img=img, c=c)    
            main_hue_range = find_main_color(isolated, all_hue_range) 
            rgb_class = RANGE_HUE_LABEL[main_hue_range]
            # save_path_isolated = os.path.join(out_folder, f'{label}_{ci}_{conf:.2f}.png')
            # cv2.imwrite(save_path_isolated, isolated) # isolated is BGR, yolo output is BGR
            # print ('save_path_isolated', save_path_isolated)
            # print ('main_hue_range', rgb_class)
            if conf > final_res[target_class][0]:
                  final_res[target_class][0] = conf
                  final_res[target_class][1] = rgb_class
    print (rgb_class)
