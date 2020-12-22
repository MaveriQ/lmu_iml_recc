from pytorch_lightning import LightningDataModule
from torch.utils.data import random_split, DataLoader
import torch

import pandas as pd
import numpy as np
import re, os
from pathlib import Path
from torch.utils.data.dataset import TensorDataset
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
import blosc

class Netflix_DataModule(LightningDataModule):

    def __init__(self, args):
        super().__init__()
        # self.batch_size = batch_size
        self.args = args
    
    def setup(self, stage = None):

        size = 100480507
        ratings = np.empty(size,dtype=np.int8)
        movie_id = np.empty(size,dtype=np.int16)
        user_id = np.empty(size,dtype=np.int32)

        rating_file = str(self.args.data_dir/'ratings.dat')
        users_file = str(self.args.data_dir/'users.dat')
        movies_file = str(self.args.data_dir/'movies.dat')

        blosc.decompress_ptr(pa.OSFile(users_file).readall(), user_id.__array_interface__['data'][0])
        blosc.decompress_ptr(pa.OSFile(movies_file).readall(), movie_id.__array_interface__['data'][0])
        blosc.decompress_ptr(pa.OSFile(rating_file).readall(), ratings.__array_interface__['data'][0])

        rating_tensor = torch.FloatTensor(ratings)
        movie_id_tensor = torch.LongTensor(movie_id)
        user_id_tensor = torch.LongTensor(user_id)

        dataset = TensorDataset(movie_id_tensor,user_id_tensor,rating_tensor)

        training_size = int(0.9*size)
        val_test_size = size-training_size
        self.netflix_train, netflix_val_test = random_split(dataset,lengths=[training_size,val_test_size])

        val_size = int(0.5*val_test_size)
        test_size = val_test_size - val_size
        self.netflix_val, self.netflix_test = random_split(netflix_val_test,lengths=[val_size,test_size])        


    def train_dataloader(self):
        return DataLoader(self.netflix_train, batch_size=self.args.batch_size,num_workers=8,pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.netflix_val, batch_size=self.args.batch_size,num_workers=8,pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.netflix_test, batch_size=self.args.batch_size,num_workers=8,pin_memory=True)