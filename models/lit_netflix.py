from argparse import ArgumentParser

import torch
import pytorch_lightning as pl
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split

from torchvision.datasets.mnist import MNIST
from torchvision import transforms


class Netflix_Recommender_Engine(pl.LightningModule):
    def __init__(self): # , hidden_dim=128, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()

        self.movie_embedding = torch.nn.Embeddng(17770,self.hparams.embedding_dim)
        self.user_embedding = torch.nn.Embeddng(480189,self.hparams.embedding_dim)
        self.movie_l1 = torch.nn.Linear(self.hparams.embedding_dim, self.hparams.hidden_dim)
        self.user_l1 = torch.nn.Linear(self.hparams.embedding_dim, self.hparams.hidden_dim)
        self.l2 = torch.nn.Linear(self.hparams.hidden_dim*2, 5)

    def forward(self, x):
        movie_id,user_id = x
        movie_vec = self.movie_embedding(movie_id)
        user_vec = self.user_embedding(user_id)

        movie_l1 = torch.relu(self.movie_l1(movie_vec))
        user_l1 = torch.relu(self.user_l1(user_vec))

        combined = torch.cat([movie_l1,user_l1])
        rating = self.l2(combined)
        return rating

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log('valid_loss', loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log('test_loss', loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--embedding_dim', type=int, default=128)
        parser.add_argument('--hidden_dim', type=int, default=256)
        parser.add_argument('--learning_rate', type=float, default=0.0001)
        return parser


def cli_main():
    pl.seed_everything(1234)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--batch_size', default=32, type=int)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = Netflix_Recommender_Engine.add_model_specific_args(parser)
    args = parser.parse_args()

    # ------------
    # data
    # ------------
    dataset = MNIST('', train=True, download=True, transform=transforms.ToTensor())
    mnist_test = MNIST('', train=False, download=True, transform=transforms.ToTensor())
    mnist_train, mnist_val = random_split(dataset, [55000, 5000])

    train_loader = DataLoader(mnist_train, batch_size=args.batch_size)
    val_loader = DataLoader(mnist_val, batch_size=args.batch_size)
    test_loader = DataLoader(mnist_test, batch_size=args.batch_size)

    # ------------
    # model
    # ------------
    model = Netflix_Recommender_Engine()#(args.hidden_dim, args.learning_rate)

    # ------------
    # training
    # ------------
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, train_loader, val_loader)

    # ------------
    # testing
    # ------------
    trainer.test(test_dataloaders=test_loader)


if __name__ == '__main__':
    cli_main()
