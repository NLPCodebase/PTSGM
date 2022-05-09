import pandas as pd


data = pd.read_csv("D:/data/Causal-TimeBank-main/event_causal_timebank.csv", sep=',',encoding = 'UTF-8').values.tolist()
df = pd.DataFrame(data, columns=["Source sentence","Answer sentence","Event1","Event2","labels"])
train_df=df.sample(frac=0.8)
eval_df=df[~df.index.isin(train_df.index)]

train_outputpath = './data/train_causal_timebank.csv'
eval_outputpath = './data/test_causal_timebank.csv'

train_df.to_csv(train_outputpath,sep=',',encoding = 'UTF-8',index=False,header=True) 
eval_df.to_csv(eval_outputpath,sep=',',encoding = 'UTF-8',index=False,header=True) 
