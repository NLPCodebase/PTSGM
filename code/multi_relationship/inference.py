import torch
import time
import math
import csv
import logging

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score

from itertools import islice

from transformers import BartForConditionalGeneration, BartTokenizer, BartConfig


from seq2seq_model import Seq2SeqModel,AutoTokenizer


class InputExample():
    def __init__(self, input_TXT, event1, event2, labels):
        self.input_TXT = input_TXT
        self.event1 = event1
        self.event2 = event2
        self.labels = labels


def predict_relation(input_TXT, event1, event2):  # 预测一个句子中两个事件的关系
    input_TXT = [input_TXT]*3
    input_ids = tokenizer(input_TXT, return_tensors='pt')['input_ids']
    model.to(device)
    template_list = ["原因事件。", "后续事件。", "无关事件。"]
    relation_dict = {0: '因果关系', 1: '顺承关系', 2: 'NONE'}
    temp_list = []
    for k in range(len(template_list)):
        temp_list.append(event1+"是"+event2+"的"+template_list[k])

    output_ids = tokenizer(temp_list, return_tensors='pt',
                           padding=True, truncation=True)['input_ids']
    # 加一个unused字符
    # output_ids[:, 0] = 2
    output_length_list = [0]*3

    base_length = ((tokenizer(temp_list[2], return_tensors='pt', padding=True, truncation=True)[
                   'input_ids']).shape)[1] - 4
    output_length_list[0:2] = [base_length]*3

    score = [1]*3
    with torch.no_grad():
        
        output = model(input_ids=input_ids.to(device), decoder_input_ids=output_ids[:, :output_ids.shape[1] - 2].to(device))[0]
        # print(tokenizer.decode(output_ids[2, :output_ids.shape[1] - 4]))
        for i in range(output_ids.shape[1] - 3):
        # output = model(input_ids=input_ids.to(device), decoder_input_ids=output_ids.to(device))[0]
        # for i in range(output_ids.shape[1] - 1):
            # print(input_ids.shape)
            logits = output[:, i, :]
            logits = logits.softmax(dim=1)
            # values, predictions = logits.topk(1,dim = 1)
            logits = logits.to('cpu').numpy()
            # print(output_ids[:, i+1].item())
            for j in range(0, 3):
                if i < output_length_list[j]:
                    score[j] = score[j] * logits[j][int(output_ids[j][i + 1])]

    return relation_dict[(score.index(max(score)))]


def cal_time(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

TokenModel = "./exp/template"
tokenizer = AutoTokenizer.from_pretrained(TokenModel)
# input_TXT = "Japan began the defence of their Asian Cup title with a lucky 2-1 win against Syria in a Group C championship match on Friday ."

model = BartForConditionalGeneration.from_pretrained('./exp/template')
# model = BartForConditionalGeneration.from_pretrained('../dialogue/bart-large')
model.eval()
model.config.use_cache = False
# input_ids = tokenizer(input_TXT, return_tensors='pt')['input_ids']
# print(input_ids)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
examples = []


f = open('./data/test.csv', 'r')
with f:
    reader = csv.reader(f)
    for row in islice(reader, 1, None):
        input_TXT = row[0]
        event1 = row[2]
        event2 = row[3]
        labels = row[4]
        examples.append(InputExample(input_TXT=input_TXT, event1=event1, event2=event2, labels=labels))


trues_list = []
preds_list = []
num_01 = len(examples)
num_point = 0
start = time.time()
for example in examples:
    preds_list.append(predict_relation(example.input_TXT,
                      example.event1, example.event2))
    trues_list.append(example.labels)
    print('%d/%d (%s)' % (num_point+1, num_01, cal_time(start)))
    print('Pred:', preds_list[num_point])
    print('Gold:', trues_list[num_point])
    num_point += 1
print(classification_report(trues_list,preds_list))
class_label=['No relationship','Cause-Effect','Follow-up']
conf_mat = confusion_matrix(trues_list, preds_list)
df_cm = pd.DataFrame(conf_mat, index=class_label, columns=class_label)
heatmap = sns.heatmap(df_cm, annot=True, fmt='d', cmap='YlGnBu')
heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right')
heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right')


plt.ylabel("true label")
plt.xlabel("predict label")
plt.show()
results = {
    "F": f1_score(trues_list, preds_list,average=None),
    "P": precision_score(trues_list, preds_list,average=None),
    "R": recall_score(trues_list, preds_list,average=None)
}
print(results)
for num_point in range(len(preds_list)):
    preds_list[num_point] = ' '.join(preds_list[num_point]) + '\n'
    trues_list[num_point] = ' '.join(trues_list[num_point]) + '\n'
with open('./pred.txt', 'w') as f0:
    f0.writelines(preds_list)
with open('./gold.txt', 'w') as f0:
    f0.writelines(trues_list)
'''
count = 0
sum = 0
for lable1,lable2 in  zip(preds_list,trues_list):
    if lable1==lable2:
        count +=1
        sum +=1
    else:
        sum +=1
print(count,sum,count/sum)
print(classification_report(trues_list,preds_list))
cm = confusion_matrix(trues_list,preds_list)
sns.heatmap(cm,cmap = sns.color_palette("Blues"),annot = True)


plt.ylabel("true label")
plt.xlabel("predict label")
plt.show()

results = {
    "F": f1_score(trues_list, preds_list,average='weighted'),
    "P": precision_score(trues_list, preds_list,average='weighted'),
    "R": recall_score(trues_list, preds_list,average='weighted')
    # average has to be one of (None, 'micro', 'macro', 'weighted', 'samples')
    # "F": f1_score(trues_list, preds_list,average=None),
    # "P": precision_score(trues_list, preds_list,average=None),
    # "R": recall_score(trues_list, preds_list,average=None)
}
print(results)
for num_point in range(len(preds_list)):
    preds_list[num_point] = ' '.join(preds_list[num_point]) + '\n'
    trues_list[num_point] = ' '.join(trues_list[num_point]) + '\n'
with open('./pred.txt', 'w') as f0:
    f0.writelines(preds_list)
with open('./gold.txt', 'w') as f0:
    f0.writelines(trues_list)

'''
