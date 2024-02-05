import yaml
import os
import argparse
from dvclive.lightning import DVCLiveLogger
# from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.trainer.trainer import Trainer
from model import Net
from datamodule import LungSegmentationDataModule

parser = argparse.ArgumentParser()
parser.add_argument('--params', '-p', type = str, help = 'params file', required = True)

def main():

    args = parser.parse_args()

    params = yaml.safe_load(open(args.params))

    #params = yaml.safe_load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "params.yaml")))

    live = DVCLiveLogger(save_dvc_exp = True, log_model = True, dir = "../results1") # report = "notebook", log_model=True

    print(params["network_parameter"]["input_size"])
    
    # data module
    dm = LungSegmentationDataModule(os.path.join(params["dataset"]["data_dir"],'datalist.csv'),batch_size= params["training_parameter"]["batch_size"],
                                    input_size = params["network_parameter"]["input_size"], num_workers=params["dataset"]["num_workers"])

    # Network
    model = Net(input_size=params["network_parameter"]["input_size"],
                num_classes= params["network_parameter"]["num_classes"],
                learning_rate= params["training_parameter"]["learning_rate"])

    # Trainer
    checkpoint_callback = ModelCheckpoint(dirpath="../lightning_models/", save_top_k=1, monitor="val_loss", filename='{run_name}_lungseg-{epoch:02d}-{val_loss:.2f}')

    trainer = Trainer(min_epochs=0, max_epochs= params["training_parameter"]["num_epochs"], accelerator = params["compute"]["accelerator"], callbacks=[checkpoint_callback],
                        logger = live, log_every_n_steps = 5)

    trainer.fit(model,dm)

if __name__ == "__main__":

    main()
    