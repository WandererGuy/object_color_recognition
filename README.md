python main.py

```
python upload_main.py
```




<!-- # Object Color Recognition

## Description
This project focuses on recognizing the predominant color (hue) of objects using computer vision based on the color wheel theory. Each object is classified by a specific hue range, making use of the **hue** component from the HSV color space for more accurate color detection than BGR. 

The program defines 12 distinct hue ranges according to color wheel theory, along with an additional range for **black, grey, and white**.

### Advantages of Hue over BGR:
- **Hue** is more accurate in representing true colors of objects.
- BGR might misinterpret lighting and other environmental factors that affect object color perception.

### Limitations:
- If a colorful light shines on the object, the program might detect the light color rather than the object's actual color.
- Too much shadow on the object could result in detection of **black**.
- Objects with faded colors or excess light may be misidentified as **white**.

---

## prerequisites 
given a result image with known/ interested target class in it (example: a yolo bounding box crop image and its class)


## Environment Setup

To set up the environment, refer to the instructions provided in `prepare.md` for a complete guide on installing dependencies and configuring the environment.

---

## Usage

### Scripts Overview

1. **create_rgb.py**:
   - This script generates the RGB representation for a given range of hues.
   - **Usage**: `python create_rgb.py`
   - result stored in folder rgb_for_hue_range, only black_grey_white dont have rgb cause im lazy lol

2. **infer.py**:
   - This script performs inference on images, detecting the hue-based color of objects.
   - **Usage**: `python infer.py`

3. **main_with_segment.py**:
   - This script sets up a FastAPI server to expose the inference functionality through an API.
   first fix your own ip in config.yml file
   - **Usage**: `python main_with_segment.py`
   this will turn on the server with API for upcoming request , u can request by postman 
   (it segment the interest object class inside image with the object and recognise the object hue color range)
   
   using postman request , with form-data in body , with 2 keys , target_class and img_path
---

## API Usage

Once the FastAPI server is running using `main.py`, you can interact with the inference functionality via API endpoints.

---

Feel free to extend or modify the code based on your specific use cases. -->
