import pandas as pd
import numpy as np
import re, os
from pathlib import Path
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path


# In bash run ```$ cat *.txt>>combined.txt``` to concatenate all movie rating files into one


rating_file = Path('../../data/training_set/combined.txt')

print('Reading data file....\n')
with open(rating_file,'r',encoding='utf-8',errors='ignore') as f:
    data = f.readlines()

print('Extracting data...\n')
rating_data=[]
for i,d in tqdm(enumerate(data),total=len(data)):
    d=d.strip()
    f=re.search('\d+:',d)
    if f is not None:
        movie = d[:-1]
        continue
    rating=d.split(',')
    rating_data.append((movie,*rating))
rating_df = pd.DataFrame(rating_data,columns=['movie_id','user_id','rating','date'])

table = pa.Table.from_pandas(rating_df)
pq.write_table(table, 'rating.parquet')