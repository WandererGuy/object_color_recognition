def collect_pseudo_data(image_path):
    import requests

    url = "http://192.168.1.8:3013/detect-recognize-color"

    payload = {'image_path': image_path}
    files=[

    ]
    headers = {}

    response = requests.request("POST", url, headers=headers, data=payload, files=files)




import os 



# Define the supported image file extensions
image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}



def find_images_in_folder(root_folder):
    max_count = 25000
    count = 0
    image_files = []
    for dirpath, dirnames, filenames in os.walk(root_folder):
        for filename in filenames:
            # Check if the file has an image extension
            if any(filename.lower().endswith(ext) for ext in image_extensions):
                # Get the full path of the image file
                image_files.append(os.path.join(dirpath, filename))
                count += 1
                if count >= max_count:
                    return image_files
    return image_files

root_folder = r"D:\WORK\OBJECT_COLOR\Vehicle_dataset_2\archive\daytime-dataset"
image_files = find_images_in_folder(root_folder)

import random


random.seed(42)

# Shuffle the list in place
random.shuffle(image_files)

from tqdm import tqdm
for image_file in tqdm(image_files, total=len(image_files)):
    collect_pseudo_data(image_file)