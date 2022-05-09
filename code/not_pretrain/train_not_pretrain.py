


import pandas as pd
import logging

from seq2seq_model_not_pretrain import Seq2SeqModel

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)


# data = pd.read_csv("./data/train.csv", sep=',',encoding = 'gbk',usecols=[0,1]).values.tolist()
data = pd.read_csv("./data/train_not_pretrain.csv", sep=',',encoding = 'UTF-8',usecols=[0,1]).values.tolist()
df = pd.DataFrame(data, columns=["input_text", "target_text"])
train_df=df.sample(frac=0.8)#按0.8比例随机采样
print(len(train_df))
eval_df=df[~df.index.isin(train_df.index)]


model_args = {
    "reprocess_input_data": True,
    "overwrite_output_dir": True,
    "max_seq_length": 64,
    "train_batch_size": 32,
    "num_train_epochs": 2,
    "save_eval_checkpoints": False,
    "save_model_every_epoch": False,
    "evaluate_during_training": True,
    "evaluate_generated_text": True,
    "evaluate_during_training_verbose": True,
    "use_multiprocessing": False,
    "max_length": 64,
    "manual_seed": 4,
    "save_steps": 11898,
    "gradient_accumulation_steps": 1,
    "output_dir": "./exp/notpretrain",
}

# Initialize model
model = Seq2SeqModel(
    encoder_decoder_type="bart",
    encoder_decoder_name="D:/bart-base",
    # encoder_decoder_name="./exp/notpretrain",
    args=model_args,
    # use_cuda=False,
)



# Train the model
model.train_model(train_df, eval_data=eval_df)

# Evaluate the model
results = model.eval_model(eval_df)

# Use the model for prediction

print(model.predict(["During Lohan's stay at the famed Betty Ford Center in 2010, Lohan was accused of attacking a female staffer"]))
