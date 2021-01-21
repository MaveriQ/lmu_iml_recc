from argparse import ArgumentParser

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from torch.nn import functional as F
from pathlib import Path
import pdb

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from src.pl_dm_netflix import Netflix_DataModule

class Netflix_Recommender_Engine_Conv1D(pl.LightningModule):
    def __init__(self,hparams): # , hidden_dim=128, learning_rate=1e-3):
        super().__init__()
        self.hparams = hparams
        self.save_hyperparameters()

        self.movie_embedding = torch.nn.Embedding(1000,self.hparams.embedding_dim) # 17770
        self.user_embedding = torch.nn.Embedding(10000,self.hparams.embedding_dim) # 480189

        self.movie_l1 = torch.nn.Conv1d(in_channels=1, out_channels=self.hparams.hidden_dim, kernel_size=self.hparams.kernel_size,padding=self.hparams.kernel_size//2)
        self.user_l1 = torch.nn.Conv1d(in_channels=1, out_channels=self.hparams.hidden_dim, kernel_size=self.hparams.kernel_size,padding=self.hparams.kernel_size//2)

        # self.movie_l1 = torch.nn.Linear(self.hparams.embedding_dim, self.hparams.hidden_dim)
        # self.user_l1 = torch.nn.Linear(self.hparams.embedding_dim, self.hparams.hidden_dim)

        # if self.hparams.is_classifer:
            # print('Classification loss')
        self.l2 = torch.nn.Linear(self.hparams.hidden_dim**2, 5)
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

        movie_l1 = torch.relu(self.movie_l1(movie_vec.unsqueeze(1)))
        user_l1 = torch.relu(self.user_l1(user_vec.unsqueeze(1)))

        combined = torch.cat([movie_l1,user_l1],dim=2)
        rating = self.l2(combined.reshape(self.hparams.batch_size,-1))
        return rating

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
        optim = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim,
                                                            patience=2,
                                                            verbose=True)
        return {'optimizer':optim,
                'lr_scheduler':sched,
                'monitor':'valid_acc_epoch'}

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--embedding_dim', type=int, default=128)
        parser.add_argument('--hidden_dim', type=int, default=256)
        parser.add_argument('--kernel_size', type=int, default=3)
        parser.add_argument('--lr', type=float, default=0.001)
        return parser

class Netflix_Recommender_Engine_Linear(pl.LightningModule):
    def __init__(self,hparams): # , hidden_dim=128, learning_rate=1e-3):
        super().__init__()
        self.hparams = hparams
        self.save_hyperparameters()

        self.movie_embedding = torch.nn.Embedding(1000,self.hparams.embedding_dim) # 17770
        self.user_embedding = torch.nn.Embedding(10000,self.hparams.embedding_dim) # 480189
        self.movie_l1 = torch.nn.Linear(self.hparams.embedding_dim, self.hparams.hidden_dim)
        self.user_l1 = torch.nn.Linear(self.hparams.embedding_dim, self.hparams.hidden_dim)

        # if self.hparams.is_classifer:
            # print('Classification loss')
        self.l2 = torch.nn.Linear(self.hparams.hidden_dim**2, 5)
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

        movie_l1 = torch.relu(self.movie_l1(movie_vec.unsqueeze(1)))
        user_l1 = torch.relu(self.user_l1(user_vec.unsqueeze(1)))

        combined = torch.cat([movie_l1,user_l1],dim=2)
        rating = self.l2(combined.reshape(self.hparams.batch_size,-1))
        return rating

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
        # self.log('valid_loss', loss)

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
        optim = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim,
                                                            mode='max',
                                                            patience=2,
                                                            verbose=True)
        return {'optimizer':optim,
                'scheduler':sched,
                'monitor': 'valid_acc_epoch'}

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--embedding_dim', type=int, default=128)
        parser.add_argument('--hidden_dim', type=int, default=256)
        parser.add_argument('--lr', type=float, default=0.001)
        return parser


def cli_main():
    pl.seed_everything(1234)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--exp_name', default='lmu_recc_iml', type=str)
    # parser.add_argument('--is_classifer',action='store_true')
    # parser.add_argument('--fast_dev_run',action='store_true')

    parser = pl.Trainer.add_argparse_args(parser)
    parser = Netflix_Recommender_Engine_Linear.add_model_specific_args(parser)

    # debug_args = "--fast_dev_run --gpus 1 --accelerator ddp".split()
    debug_args = "--gpus 1 --accelerator ddp --batch_size 4096 --max_epochs 10 --lr 1e-2 --exp_name conv_lr_1e2_bs_4096_test --val_check_interval 0.01 --limit_val_batches 0.01".split()
    args = parser.parse_args()

    # ------------
    # data
    # ------------
    args.data_dir = Path('../data/')
    dm = Netflix_DataModule(args)

    # ------------
    # callbacks
    # ------------

    lr_logger = LearningRateMonitor(logging_interval='step')

    # ------------
    # model
    # ------------
    model = Netflix_Recommender_Engine_Linear(args)#(args.hidden_dim, args.learning_rate)

    # ------------
    # logger
    # ------------

    tb_logger = pl.loggers.TensorBoardLogger(
                                    save_dir=os.getcwd(),
                                    version=args.exp_name,
                                    name='lightning_tb_logs'
                                    )
    # ------------
    # training
    # ------------
    trainer = pl.Trainer.from_argparse_args(args,
                                            logger=[tb_logger])
    trainer.fit(model, dm)

    # ------------
    # testing
    # ------------
    # trainer.test(test_dataloaders=test_loader)


if __name__ == '__main__':
    cli_main()