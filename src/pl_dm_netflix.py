from argparse import ArgumentParser
from pytorch_lightning import LightningDataModule
from torch.utils.data import random_split, DataLoader
import torch

import numpy as np
from pathlib import Path
from torch.utils.data.dataset import TensorDataset
import pyarrow as pa
from pathlib import Path
import blosc

class Netflix_DataModule(LightningDataModule):

    def __init__(self, args):
        super().__init__()
        # self.batch_size = batch_size
        self.args = args
    
    def setup(self, stage = None):

        size = 100480507 # 6491181
        ratings = np.empty(size,dtype=np.int8)
        movie_id = np.empty(size,dtype=np.int16)
        user_id = np.empty(size,dtype=np.int32)

        rating_file = str(self.args.data_dir/'ratings.dat')
        users_file = str(self.args.data_dir/'users.dat')
        movies_file = str(self.args.data_dir/'movies.dat')

        blosc.decompress_ptr(pa.OSFile(users_file).readall(), user_id.__array_interface__['data'][0])
        blosc.decompress_ptr(pa.OSFile(movies_file).readall(), movie_id.__array_interface__['data'][0])
        blosc.decompress_ptr(pa.OSFile(rating_file).readall(), ratings.__array_interface__['data'][0])

        rating_tensor = torch.LongTensor(ratings)
        movie_id_tensor = torch.LongTensor(movie_id)
        user_id_tensor = torch.LongTensor(user_id)

        rating_tensor = rating_tensor - 1

        dataset = TensorDataset(movie_id_tensor,user_id_tensor,rating_tensor)

        training_size = int(0.9*size)
        val_test_size = size-training_size
        self.netflix_train, netflix_val_test = random_split(dataset,lengths=[training_size,val_test_size])

        val_size = int(0.5*val_test_size)
        test_size = val_test_size - val_size
        self.netflix_val, self.netflix_test = random_split(netflix_val_test,lengths=[val_size,test_size])        


    def train_dataloader(self):
        return DataLoader(self.netflix_train, batch_size=self.args.batch_size,num_workers=8,pin_memory=True,drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.netflix_val, batch_size=self.args.batch_size,num_workers=8,pin_memory=True,drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.netflix_test, batch_size=self.args.batch_size,num_workers=8,pin_memory=True,drop_last=True)

if __name__=="__main__":
    parser = ArgumentParser()
    args = parser.parse_args()
    args.data_dir = Path('./data/')
    dm = Netflix_DataModule(args)
    dm.setup()
    print(dm.netflix_train)