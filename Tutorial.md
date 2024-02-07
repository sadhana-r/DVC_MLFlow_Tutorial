# DVC MLOps tutorial

## Set up python environment

```

conda env create --name mlops_env
pip install -r requirements.txt

```

## If starting a new project:

### Initialize DVC

https://dvc.org/doc/command-reference/init

```
git init
dvc init

```

### Commit dvc files to git

```
git add .dvc/.gitignore
git add .dvc/config
git add .dvcignore
git commit -m "Initialize dvc"
```

### Add the data to dvc tracking

```
dvc add data/..
```

Then add the data/*.dvc files to git tracking

### Push the data to a gdrive remote folder

Need to fitst install the dvc library for the remote server

```
pip install dvc_gdrive
```
```
dvc remote add --default drive gdrive://<Folder ID>
dvc remote modify drive gdrive_acknowledge_abuse true
```
This will ask you to authorize your google account access and save your credentials to a gdrive_credentials.json. 
The .dvc/config file gets updated to reflect the remote directory. 

Push the data to the remote directory

```
dvc push
```

## On another machine/remote server

###  Git clone the DVC repository and pull the data

Pull the data:

```
dvc pull
```

If not able to authorize accessvia the internet, you can point dvc remote to the location of the credential file:
```
dvc remote modify gdrive gdrive_user_credentials_file ..\gdrive_credentials.json

```

## DVC pipelines and experiment tracking

## Add stages to a dvc.yaml file

### Option 1: Through the command line
```
dvc stage add --name preprocess 
--deps data/MontgomerySet --deps data/ChinaSet_AllFiles 
--outs data/datalist.csv
python src/pipline/preprocess.py

```

### Option 2: Manually add to dvc.yaml
```
preprocess:
  cmd: python src/pipeline/preprocess.py
  deps:
  - data/MontgomerySet
  - data/ChinaSet_AllFiles 
  outs:
  - data/datalist.csv
```

#### You can also add parameter dependencies

```
train:
  cmd: python src/pipeline/train_dvc.py --params src/pipeline/params.yaml
  deps:
  - ./data/datalist.csv
  params:
  - ./src/pipeline/params.yaml:
    - dataset.data_dir
    - training_parameter.batch_size
    - training_parameter.learning_rate
    - network_parameter.input_size
    - network_parameter.num_classes
    - dataset.num_workers

```


## Running the pipeline

### Option 1: Without experiment tracking:
```
dvc repro
```

### Option 2: With experiment tracking

First you need to install the dvc library for experiment tracking: DVCLive

```
pip install dvclive
```

DVCLive has a logger that supports pytorch lightning

To run the experiment: 

```
dvc exp run --name NAME
dvc exp run --name --set-params training_parameter.batch_size=6

```

## Sharing experiments

```
dvc exp push [git_remote] [experiment_name] --rev [can specify commit]
```
