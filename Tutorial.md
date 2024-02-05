conda activate mlops

# Activate git and DVC
git init
dvc init

# Commit dvc files to git

git add 
.dvc/.gitignore
.dvc/config
.dvcignore

# Add the data to dvc tracking
dvc add data/..

# Add gdrive remote folder
dvc remote add --default drive gdrive://<Folder ID>
dvc remote modify drive gdrive_acknowledge_abuse true

** Check config file ** 
### On remote server
dvc remote modify gdrive gdrive_user_credentials_file ..\gdrive_credentials.json

# DVC pipeline

## Add preprocessing stage

dvc stage add --name preprocess --deps data/MontgomerySet --deps data/ChinaSet_AllFiles --outs data/datalist.csv python src/pipline/preprocess.py

## Manually add to dvc.yaml file
  train:
    cmd: python src/pipeline/train_dvc.py
    deps:
    - data/datalist.csv
    outs:
    - .ckpt

## Run the pipeline
 - with error in dependency location
  Command: dvc repro


