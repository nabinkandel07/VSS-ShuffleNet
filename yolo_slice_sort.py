import os
import random
import shutil

# After using yolo_label, you will get a bunch of images with annotations.
# This python file will separate the images and annotations into YOLO format
# Into 'Train/Valid/Test' so that we can immediately start training
#

# Set the path to the folder containing images and annotations
data_folder = "D:/Download/NEU-Check13/NEU-DET/IMAGES"

# Set the path to the output folders
train_folder = "D:/Download/NEU-Check13/train"
val_folder = "D:/Download/NEU-Check13/valid"

# Set the split ratios
train_ratio = 0.7
val_ratio = 0.3

# Set the file extension
file_extension = ".jpg"  # Change this to ".jpg" or anything else if needed

# =========================

# Create the output folders if they don't exist
os.makedirs(os.path.join(train_folder, "images"), exist_ok=True)
os.makedirs(os.path.join(train_folder, "labels"), exist_ok=True)
os.makedirs(os.path.join(val_folder, "images"), exist_ok=True)
os.makedirs(os.path.join(val_folder, "labels"), exist_ok=True)

# Get the list of all files in the data folder
all_files = os.listdir(data_folder)

# Filter and get the list of image files
image_files = [filename for filename in all_files if filename.endswith(file_extension)]

# Check if any image files found
if len(image_files) == 0:
    print(f"No {file_extension} files found in the specified folder. Please ensure the files are in the correct format.")
    exit()
else:
    print(f"Number of {file_extension} files found:", len(image_files))

# Filter and get the list of annotation files
annotation_files = [filename for filename in all_files if filename.endswith(".txt")]

# Check if any annotation files found
if len(annotation_files) == 0:
    print("No annotation files found.")
    exit()
else:
    print("Number of annotation files found:", len(annotation_files))

# Shuffle the image files
random.shuffle(image_files)

# Calculate the number of images for each split
num_images = len(image_files)
num_train = int(num_images * train_ratio)
num_val = int(num_images * val_ratio)

# Split the image files into train, val, and test sets
train_files = image_files[:num_train]
val_files = image_files[num_train:num_train + num_val]

# Move the image files to the corresponding output folders
for file in train_files:
    image_path = os.path.join(data_folder, file)
    annotation_path = os.path.join(data_folder, file.replace(file_extension, ".txt"))
    shutil.move(image_path, os.path.join(train_folder, "images", file))
    shutil.move(annotation_path, os.path.join(train_folder, "labels", file.replace(file_extension, ".txt")))

for file in val_files:
    image_path = os.path.join(data_folder, file)
    annotation_path = os.path.join(data_folder, file.replace(file_extension, ".txt"))
    shutil.move(image_path, os.path.join(val_folder, "images", file))
    shutil.move(annotation_path, os.path.join(val_folder, "labels", file.replace(file_extension, ".txt")))

print("Data splitting completed.")
