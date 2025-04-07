import cv2
import os

# Set the folder containing images and output video name
image_folder = 'demo'
video_name = 'video_demo.mp4'
t = {}
# Collect image filenames (ensure they are sorted in the right order)
for img in os.listdir(image_folder):
    if img.endswith('.jpg') or img.endswith('.png'):
        t[img] = int(os.path.basename(img).split('.')[0].split('_')[-1])
# Sorting by the integer values (in ascending order)
sorted_items = sorted(t.items(), key=lambda item: item[1])
sorted_dict = dict(sorted_items)
images = list(sorted_dict.keys())
# Read the first image to get the frame dimensions
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, _ = frame.shape

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(video_name, fourcc, 24, (width, height))

for image in images:
    img_path = os.path.join(image_folder, image)
    frame = cv2.imread(img_path)
    video.write(frame)

# Release the VideoWriter
video.release()
print("Video creation complete!")
