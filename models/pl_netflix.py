from argparse import ArgumentParser

import torch
import pytorch_lightning as pl
from torch.nn import functional as F
from pathlib import Path
import pdb

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from src.pl_dm_netflix import Netflix_DataModule

class Netflix_Recommender_Engine(pl.LightningModule):
    def __init__(self,hparams): # , hidden_dim=128, learning_rate=1e-3):
        super().__init__()
        self.hparams = hparams
        self.save_hyperparameters()

        self.movie_embedding = torch.nn.Embedding(17770,self.hparams.embedding_dim)
        self.user_embedding = torch.nn.Embedding(480189,self.hparams.embedding_dim)
        self.movie_l1 = torch.nn.Linear(self.hparams.embedding_dim, self.hparams.hidden_dim)
        self.user_l1 = torch.nn.Linear(self.hparams.embedding_dim, self.hparams.hidden_dim)

        # if self.hparams.is_classifer:
            # print('Classification loss')
        self.l2 = torch.nn.Linear(self.hparams.hidden_dim*2, 6)
        self.loss = torch.nn.CrossEntropyLoss()
        # else:
        #     print('Regression loss')
        #     self.l2 = torch.nn.Linear(self.hparams.hidden_dim*2, 1)
        #     self.loss = torch.nn.MSELoss()
        self.train_accuracy = pl.metrics.Accuracy()
        self.valid_accuracy = pl.metrics.Accuracy()

    def forward(self, x):
        movie_id,user_id = x
        movie_vec = self.movie_embedding(movie_id)
        user_vec = self.user_embedding(user_id)

        movie_l1 = torch.relu(self.movie_l1(movie_vec))
        user_l1 = torch.relu(self.user_l1(user_vec))

        combined = torch.cat([movie_l1,user_l1],dim=1)
        rating = self.l2(combined)
        return rating.squeeze()

    def training_step(self, batch, batch_idx):

        loss, y_pred, y = self._step(batch)
        self.log('train_acc_step', self.train_accuracy(y_pred, y))
        self.log('trg_loss', loss)
        return loss

    def training_epoch_end(self, outs):
        # log epoch metric
        self.log('train_acc_epoch', self.train_accuracy.compute())

    def validation_step(self, batch, batch_idx):

        loss, y_pred, y = self._step(batch)
        self.log('valid_acc_step', self.valid_accuracy(y_pred, y))
        self.log('valid_loss', loss)

    def validation_epoch_end(self, outs):
        # log epoch metric
        self.log('valid_acc_epoch', self.valid_accuracy.compute())

    def test_step(self, batch, batch_idx):

        loss, y_pred, y = self._step(batch)
        self.log('test_acc_step', self.accuracy(y_pred, y))
        self.log('test_loss', loss)

    def _step(self,batch):

        movie_id,user_id,rating = batch

        x=(movie_id,user_id)
        
        # if self.hparams.is_classifer:
        y=rating.long()
        # else:
            # y=rating

        y_pred = self(x)
        loss = self.loss(y_pred, y)
        return loss, y_pred, y

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim,
                                                            patience=2,
                                                            verbose=True)
        return {'optimizer':optim,
                'scheduler':sched}

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--embedding_dim', type=int, default=128)
        parser.add_argument('--hidden_dim', type=int, default=256)
        parser.add_argument('--learning_rate', type=float, default=0.001)
        return parser


def cli_main():
    pl.seed_everything(1234)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--batch_size', default=32, type=int)
    # parser.add_argument('--is_classifer',action='store_true')
    # parser.add_argument('--fast_dev_run',action='store_true')

    parser = pl.Trainer.add_argparse_args(parser)
    parser = Netflix_Recommender_Engine.add_model_specific_args(parser)
    args = parser.parse_args()

    # ------------
    # data
    # ------------
    args.data_dir = Path('./data/')
    dm = Netflix_DataModule(args)

    # ------------
    # model
    # ------------
    model = Netflix_Recommender_Engine(args)#(args.hidden_dim, args.learning_rate)

    # ------------
    # training
    # ------------
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, dm)

    # ------------
    # testing
    # ------------
    # trainer.test(test_dataloaders=test_loader)


if __name__ == '__main__':
    cli_main()
