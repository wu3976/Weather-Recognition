# ------------------------DATA PREPERATION--------------------------#
# The dataset is not organized initially, and the images have different size.
# Therefore, resizing and labeling is needed.
# Run this only if data is not organized.
#
# @author Chenjie Wu
import os, re
from PIL import Image

ROOT_DIR = "./data/weather_dataset"
INTEND_IMG_SIZE = (256, 256)


# resize the files and store them into correspounding directories.

# Get the corresponding folder according to file_name.
def get_folder(file_name):
    search_obj = re.search(r"\d", file_name)
    after_last = search_obj.span()[0]
    return file_name[0:after_last]


# Resize the listed image and save them to correspounding folders
# @param root_dir The place to find the images.
# @param file_names LIST of images' file names.
# @param size The size of intending images. 2-TUPLE.
# @requires
#   1. All correspounding folders should already exist in the same directory of images
#   2. All images must be under root_dir.
def resize_files_and_save(root_dir, train_or_test, file_names, size):
    for file_name in file_names:
        img = Image.open(root_dir + "/" + file_name)
        resized_img = img.resize(size)
        folder = get_folder(file_name)
        resized_img.save(root_dir + "/" + train_or_test + "/" + folder + "/" + file_name)


# create label file
'''label_file = open(ROOT_DIR + "/labels.txt", "w+")'''
labels = ["cloudy", "rain", "shine", "sunrise"]
'''label_file.writelines([label + "\n" for label in labels])'''

file_names = os.listdir(ROOT_DIR)
# remove label.txt and folders
for label in labels:
    if label in file_names:
        file_names.remove(label)

if "train" in file_names:
    file_names.remove("train")

if "test" in file_names:
    file_names.remove("test")

file_names.remove("labels.txt")
# print(file_names.index("labels.txt"))

print(file_names)
# create labeling folders.
if "train" not in os.listdir(ROOT_DIR):
    os.mkdir(ROOT_DIR + "/train")

for label in labels:
    if label not in os.listdir(ROOT_DIR + "/train"):
        os.mkdir(ROOT_DIR + "/train/" + label)

resize_files_and_save(ROOT_DIR, "train", file_names, INTEND_IMG_SIZE)
print("Saved!")
