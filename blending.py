import numpy as np # NUMPY
import pandas as p # PANDAS

# DATA VIZUALIZATION LIBRARIES
from matplotlib import pyplot as plt
import seaborn as sns

# METRICS TO MEASURE RMSE
from math import sqrt
from sklearn import metrics

#ALL PUBLIC SOLUTION RMSE < 0.2269 (WITHOUT REPETITIONS)
df_base0 = p.read_csv('./input/public-solutions/base0_0.2211.csv',names=["item_id","deal_probability0"], skiprows=[0],header=None)
df_base1 = p.read_csv('./input/public-solutions/wbj0_0.2209.csv',names=["item_id","deal_probability1"], skiprows=[0],header=None)
df_base2 = p.read_csv('./input/public-solutions/wbj0_0.2239.csv',names=["item_id","deal_probability2"], skiprows=[0],header=None)
df_base3 = p.read_csv('./input/public-solutions/wbj0_0.2226.csv',names=["item_id","deal_probability3"], skiprows=[0],header=None)
df_base4 = p.read_csv('./input/public-solutions/wbj0_0.2237.csv',names=["item_id","deal_probability4"], skiprows=[0],header=None)
df_base5 = p.read_csv('./input/public-solutions/wbj0_0.2240.csv',names=["item_id","deal_probability5"], skiprows=[0],header=None)
df_base6 = p.read_csv('./input/public-solutions/wbj0_0.2248.csv',names=["item_id","deal_probability6"], skiprows=[0],header=None)
df_base7 = p.read_csv('./input/public-solutions/wbj0_0.22402.csv',names=["item_id","deal_probability7"], skiprows=[0],header=None)
df_base8 = p.read_csv('./input/public-solutions/wbj0_0.22192.csv',names=["item_id","deal_probability8"], skiprows=[0],header=None)
df_base9 = p.read_csv('./input/public-solutions/wbj0_0.22092.csv',names=["item_id","deal_probability8"], skiprows=[0],header=None)
df_base10 = p.read_csv('./input/public-solutions/wbj0_0.2237.csv',names=["item_id","deal_probability10"], skiprows=[0],header=None)
df_base11 = p.read_csv('./input/public-solutions/wbj0_0.2246.csv',names=["item_id","deal_probability11"], skiprows=[0],header=None)
df_base12 = p.read_csv('./input/public-solutions/wbj0_0.2219.csv',names=["item_id","deal_probability12"], skiprows=[0],header=None)
df_base13 = p.read_csv('./input/public-solutions/wbj0_0.2213.csv',names=["item_id","deal_probability13"], skiprows=[0],header=None)
df_base14 = p.read_csv('./input/public-solutions/base14_0.2237.csv',names=["item_id","deal_probability14"], skiprows=[0],header=None)
df_base15 = p.read_csv('./input/public-solutions/base15_0.2246.csv',names=["item_id","deal_probability15"], skiprows=[0],header=None)
df_base16 = p.read_csv('./input/public-solutions/wbj0_0.2205.csv',names=["item_id","deal_probability16"], skiprows=[0],header=None)
df_base17 = p.read_csv('./input/public-solutions/wbj0_0.2203.csv',names=["item_id","deal_probability17"], skiprows=[0],header=None)


#CREATING SOLUTIONS COLUMNS
df_base = p.merge(df_base0,df_base1,how='inner',on='item_id')
df_base = p.merge(df_base,df_base2,how='inner',on='item_id')
df_base = p.merge(df_base,df_base3,how='inner',on='item_id')
df_base = p.merge(df_base,df_base4,how='inner',on='item_id')
df_base = p.merge(df_base,df_base5,how='inner',on='item_id')
df_base = p.merge(df_base,df_base6,how='inner',on='item_id')
df_base = p.merge(df_base,df_base7,how='inner',on='item_id')
df_base = p.merge(df_base,df_base8,how='inner',on='item_id')
df_base = p.merge(df_base,df_base9,how='inner',on='item_id')
df_base = p.merge(df_base,df_base10,how='inner',on='item_id')
df_base = p.merge(df_base,df_base11,how='inner',on='item_id')
df_base = p.merge(df_base,df_base12,how='inner',on='item_id')
df_base = p.merge(df_base,df_base13,how='inner',on='item_id')
df_base = p.merge(df_base,df_base14,how='inner',on='item_id')
df_base = p.merge(df_base,df_base15,how='inner',on='item_id')
df_base = p.merge(df_base,df_base16,how='inner',on='item_id')
df_base = p.merge(df_base,df_base17,how='inner',on='item_id')

#PORTFOLIO # 0.2204 (0,1,2,14 and 18)
df_base = p.merge(df_base1,df_base4,how='inner',on='item_id')
df_base = p.merge(df_base,df_base2,how='inner',on='item_id')
df_base = p.merge(df_base,df_base11,how='inner',on='item_id')
df_base = p.merge(df_base,df_base13,how='inner',on='item_id')
df_base = p.merge(df_base,df_base14,how='inner',on='item_id')
df_base = p.merge(df_base,df_base16,how='inner',on='item_id')
