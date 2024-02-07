import lightning as L
import torch
import torch.optim as optim
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from monai.losses import DiceCELoss
from monai.networks.nets import UNETR, UNet
import torchmetrics


class Net(L.LightningModule):
    def __init__(self, input_size, num_classes, learning_rate):
        super().__init__()

        self.model = UNet(spatial_dims = 2, in_channels = 1, out_channels = num_classes, channels = (8,16,24), strides = (1,1))
        #self.model = UNETR(in_channels = 1, out_channels = num_classes, img_size = input_size, spatial_dims=2)
        self.loss = DiceCELoss(to_onehot_y=True, softmax=True)
        self.learning_rate = learning_rate
        self.train_dice = torchmetrics.classification.Dice(num_classes=2)
        self.val_dice = torchmetrics.classification.Dice(num_classes=2)

    def forward(self,x):
        x = self.model(x)
        return x

    def configure_optimizers(self):
        print(self.learning_rate)
        return optim.Adam(self.parameters(), lr=self.learning_rate)

    def training_step(self, batch, batch_idx):
        loss, pred, y = self.common_step(batch,batch_idx)
        dice = self.train_dice(pred, y)
        self.log_dict({"train_loss":loss, "train_dsc": dice}, on_epoch = True, on_step = False)
        return loss
        # can return loss, log values

    def validation_step(self, batch, batch_idx):
        # validation loigic
        loss, pred, y = self.common_step(batch,batch_idx)
        dice = self.val_dice(pred, y)
        self.log_dict({"val_loss":loss, "val_dsc": dice}, on_epoch = True, on_step = False)
        #mlflow.log_metric("val_loss", loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        loss, pred, y = self.common_step(batch,batch_idx)
        return loss, pred, y

    def common_step(self, batch, batch_idx):
        # training logic
        x = batch["image"]
        y = batch["label"].int()

        pred = self.forward(x)
        loss = self.loss(pred,y)

        return loss, pred, y
    
    def predict_step(self, batch, batch_idx):
        x = batch["image"]

        pred = self.forward(x)
        pred = pred.softmax(dim = 1) # Convert to probability map
        preds = torch.argmax(pred, dim=1)
        return preds
