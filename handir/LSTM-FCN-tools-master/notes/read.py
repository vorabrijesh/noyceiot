import pandas as pd


df = pd.read_csv('ArrowHead_TRAIN.txt', sep='\t', header=None)
print(df.head())