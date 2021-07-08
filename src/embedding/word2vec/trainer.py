from src.dataloader.word2vec_dataloader import Word2vecDataloader
from src.embedding.word2vec.model import SkipGramModel
from torch.optim import Adam
import torch.nn.functional as F
from src.trainer.embedding_base import EmbeddingBase, get_trainer, setup_earlystop
from src.util.cuda_status import get_num_gpus
import argparse
import logging
logger = logging.getLogger('trainer.NGramLanguageModeler')


def train(args):
    args.determ = True
    dl = Word2vecDataloader(args.context_size,args.nbyte)
    dl.setup()
    args.vocab_size = dl.vocab_size
    if args.no_cuda:
        args.gpus = 0
    else:
        args.gpus = get_num_gpus()
    
    logger.info(f'args: {args}')

    for i in range(dl.nfold):
        model = Word2vecModelerTrainer(args)

        early_stop = model.get_early_stop(patience=args.patience)
        trainer = get_trainer(args.gpus, args.epochs, early_stop, None)

        dl.get_fold(i)
        trainer.fit(model,train_dataloader=dl.train_dataloader,val_dataloaders=dl.val_dataloader)
        results = trainer.predict(model,dataloaders=dl.test_dataloader)

def setup_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=64,
                        help='Number of hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--determ', action='store_true', default=False,
                        help='Deterministic flag')
    parser.add_argument('--nbyte', type=int, default=1,
                        help='Deterministic flag')
    parser.add_argument('--embedding-dim', type=int, default=10,
                        help='Deterministic flag')
    parser.add_argument('--context-size', type=int, default=10,
                        help='Deterministic flag')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='Debug flag')
    parser.add_argument('--patience', type=int, default=12,
                        help='Patience for early stopping.')

    return parser.parse_args()


class Word2vecModelerTrainer(EmbeddingBase):
    def __init__(self,args):
        super().__init__()
        self.args = args
        self.model = SkipGramModel(self.args.vocab_size, self.args.embedding_dim)

    def forward(self, batch):
        pos_u, pos_v, neg_v = batch
        self.forward(pos_u, pos_v, neg_v)
        y_hat = self.model(X)
        return y_hat

    def loss(self, pred, batch):
        X, target = batch
        logger.debug('loss {} {}'.format(pred, target))
        loss = F.nll_loss(pred, target)
        return loss
    
    def metrics(self, pred, batch):
        pass

    def configure_optimizers(self):
        optimizer = Adam(self.model.parameters(),
                lr=self.args.lr, weight_decay=self.args.weight_decay)
        
        return [optimizer]

