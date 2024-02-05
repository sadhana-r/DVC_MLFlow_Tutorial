import mlflow
import yaml
import torch
import os
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.trainer.trainer import Trainer
from model import Net
from datamodule import LungSegmentationDataModule
import config

def setup_mlflow_logger():

    mlflow.set_tracking_uri(config.mlflow_uri)
    mlflow.set_experiment(config.experiment_name)
    mlflow.set_experiment_tags(config.experiment_tags)

    # Using a combination of direct mlflow logging and pytorch lightning logger
    mlflow.pytorch.autolog()

def main():

    params = yaml.safe_load(open("params.yaml"))

    setup_mlflow_logger()

    mlflow.start_run(run_name=config.run_name)

    mlf_logger = MLFlowLogger(experiment_name=mlflow.get_experiment(mlflow.active_run().info.experiment_id).name,
        tracking_uri=mlflow.get_tracking_uri(),
        run_id=mlflow.active_run().info.run_id)
    #mlflow.log_params(config)

    # data module
    dm = LungSegmentationDataModule(os.path.join(params["dataset"]["data_dir"],'datalist.csv'),batch_size= params["training_parameter"]["batch_size"],
                                    input_size = params["network_parameter"]["input_size"], num_workers=params["training_parameter"]["num_workers"])

    # Network
    model = Net(input_size=params["network_parameter"]["input_size"],
                num_classes= params["network_parameter"]["num_classes"],
                learning_rate= params["training_parameter"]["learning_rate"])

    # Trainer
    checkpoint_callback = ModelCheckpoint(dirpath="../lightning_models/", save_top_k=1, monitor="val_loss", filename='{run_name}_lungseg-{epoch:02d}-{val_loss:.2f}')

    trainer = Trainer(min_epochs=0, max_epochs=config.NUM_EPOCHS, accelerator = config.ACCELERATOR, callbacks=[checkpoint_callback],
                        logger = mlf_logger, log_every_n_steps = 5)

    trainer.fit(model,dm)

    #Log best validation checkpoint to mlflow
    mlf_logger.experiment.log_artifact(
        run_id=mlf_logger.run_id,
        local_path=checkpoint_callback.best_model_path)

    trainer.validate(model,dm)

    #trainer.test(model,dm)

    mlflow.end_run()

if __name__ == "__main__":

    main()
    