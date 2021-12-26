import pandas as pd
import shutil
import os
from shutil import move


# read in CSV file with pandas
meta_ham = pd.read_csv('metadata.csv')

for index, row in meta_ham.iterrows():
  
    # get image name and corresponding group
    img_name = row['img_id'] 
    print(img_name)

# iterate through each row of csv
