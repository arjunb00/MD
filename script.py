import pandas as pd
import shutil
import os
from shutil import move

from_folder = r"Dataset"
to_folder_base = r"Dataset2"

# read in CSV file with pandas
meta_ham = pd.read_csv("metadata.csv")

# iterate through each row of csv
for index, row in meta_ham.iterrows():
  
    # get image name and corresponding group
    img_name = row['img_id'] 
    keyword = row['diagnostic']
    # make a folder for this group, if it doesn't already exist. 
    # as long as exist_ok is True, then makedirs will do nothing if it already exists
    to_folder = os.path.join(to_folder_base, keyword)
    # move the image from its original location to this folder
    old_img_path = os.path.join(from_folder, img_name)
    new_img_path = os.path.join(to_folder, img_name)
    move(old_img_path, new_img_path)
