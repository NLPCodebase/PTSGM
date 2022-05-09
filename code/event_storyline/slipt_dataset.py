import pandas as pd


data = pd.read_csv("./data/event_train_eventstoryline.csv", sep=',',encoding = 'UTF-8').values.tolist()
df = pd.DataFrame(data, columns=["Source sentence","Answer sentence","Event1","Event2","labels"])
train_df=df.sample(frac=0.8)#按0.8比例随机采样
eval_df=df[~df.index.isin(train_df.index)]

train_outputpath = './data/train_eventstoryline.csv'
eval_outputpath = './data/test_eventstoryline.csv'
# outputpath是保存文件路径
 
train_df.to_csv(train_outputpath,sep=',',encoding = 'UTF-8',index=False,header=True) 
eval_df.to_csv(eval_outputpath,sep=',',encoding = 'UTF-8',index=False,header=True) 
