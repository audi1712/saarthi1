from sklearn.metrics import f1_score
import pickle
import yaml
import sys
import pandas as pd
import numpy as np
import tensorflow as tf

try:
  files = open(sys.argv[sys.argv.index("--config")+1],"rb")
except IndexError:
  raise FileNotFoundError
config = yaml.safe_load(files)
files.close()

test_dat = pd.read_csv(config['data_directory']+config['test_data_name'])

files = open(config['saved_model_directory'] + "tokenizer_transcription","rb")
tokenizer_transcription = pickle.load(files)
files.close()

files = open(config['saved_model_directory'] + "tokenizer_action","rb")
tokenizer_action = pickle.load(files)
files.close()
files = open(config['saved_model_directory'] + "tokenizer_object","rb")
tokenizer_object = pickle.load(files)
files.close()
files = open(config['saved_model_directory'] + "tokenizer_location","rb")
tokenizer_location = pickle.load(files)
files.close()

test_dat['transcription'] = tokenizer_transcription.texts_to_sequences(test_dat['transcription'])
test_dat['action'] = tokenizer_action.texts_to_sequences(test_dat['action'])
test_dat['object'] = tokenizer_object.texts_to_sequences(test_dat['object'])
test_dat['location'] = tokenizer_location.texts_to_sequences(test_dat['location'])

num_transcript = len(tokenizer_transcription.word_index)
num_action = len(tokenizer_action.word_index)
num_object = len(tokenizer_object.word_index)
num_location = len(tokenizer_location.word_index)
transcription_max_len = config['input_max_len']

X_test = tf.keras.preprocessing.sequence.pad_sequences(test_dat['transcription'],transcription_max_len)
Y1_test = tf.keras.utils.to_categorical([i[0]-1 for i in test_dat['action']],num_action)
Y2_test = tf.keras.utils.to_categorical([i[0]-1 for i in test_dat['object']],num_object)
Y3_test = tf.keras.utils.to_categorical([i[0]-1 for i in test_dat['location']],num_location)

model = tf.keras.models.load_model(config['saved_model_directory']+ config['saved_model_name'])
Y_pred = model.predict(X_test)
Y_true = [Y1_test,Y2_test,Y3_test]
print("F1 Score: ",f1_score(Y_true[0],np.rint(Y_pred[0]),average = 'macro')," ",f1_score(Y_true[1],np.rint(Y_pred[1]),average = 'macro')," ",f1_score(Y_true[2],np.rint(Y_pred[2]),average = 'macro'))