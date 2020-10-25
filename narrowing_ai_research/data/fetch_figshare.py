import os
import re
import yaml
import logging
import urllib.request
from zipfile import ZipFile
import narrowing_ai_research

project_dir = narrowing_ai_research.project_dir

with open(f"{project_dir}/model_config.yaml", "r") as infile:
    materials = yaml.safe_load(infile)["input_files"]

ENDPOINT_URL = "https://api.figshare.com/v2/file/download/"

# For each destination in figshare
for k,v in materials.items():

    target_path = os.path.join(project_dir, k)
    
    print(target_path)

    for f in v:
        logging.info(f)

        file_link = ENDPOINT_URL + str(f)

        # Extract the filename to check if it's zipped or not
        h = urllib.request.urlopen(file_link)
        filename = h.headers.get_filename()

        if re.sub('.zip','',filename) in os.listdir(target_path):
            logging.info(f"already collected {f}")
            continue

        # If it is zipped we download the data and extract it
        if "zip" in filename:
            logging.info(f"retrieving {str(f)}")

            temp, h = urllib.request.urlretrieve(file_link)

            logging.info(f"extracting {str(f)}")
            with ZipFile(temp, "r") as infile:
                infile.extractall(target_path)
        else:
            # If it isn't zipped we retrieve and save the data directly
            logging.info(f"retrieving and saving {str(f)}")
            urllib.request.urlretrieve(file_link, os.path.join(target_path, 
                                                               filename))
