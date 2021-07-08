import pytorch_lightning as pl
from torch.optim import Adam
from torchvision.models import vgg11

class VGGTrainer(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = vgg11()

    def forward(self, x):
        x = self.model(x)
        return 

    def configure_optimizers(self):
        optimizer = Adam(self.model.parameters(),
                lr=self.args.lr, weight_decay=self.args.weight_decay)
        
        return {'optimizer':optimizer}
