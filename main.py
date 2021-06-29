import collections
import pathlib
import re
import string
import sys
from itertools import cycle
#import tensorflow_addons as tfa

import pandas as pd
import tensorflow as tf
import os

from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import preprocessing
from tensorflow.keras import regularizers
from tensorflow.keras import utils
import seaborn as sns
import numpy as np

from sklearn.metrics import confusion_matrix, plot_confusion_matrix, classification_report, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score

from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import matplotlib.pyplot as plt
import keras.backend as K
def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall
def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


plt.rcParams['figure.figsize'] = (12, 10)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
#import tensorflow_datasets as tfds
VOCAB_SIZE = 10000#5852#10000



MAX_SEQUENCE_LENGTH = 270#250
def extract():
    dataset = pathlib.Path(".keras")

    train_dir_ack = dataset / 'train_extract'/'error_based'
    extract_ack=pd.read_csv(train_dir_ack/'error_based.csv', engine ='python')
    where_to_exstract=dataset/'train'/'error_based'
    target=extract_ack.pop('Info')
    count=0
    for i in target:
        f = open(format(where_to_exstract)+"\\"+format(count)+'.txt', 'w')
        f.write(i)
        count=count+1
        f.close()

    train_dir_http = dataset / 'train_extract' / 'boolean_based_blind'
    extract_http = pd.read_csv(train_dir_http / 'blind.csv', engine ='python')
    where_to_exstract = dataset / 'train' / 'boolean_based_blind'
    target=extract_http.pop('Info')
    count=0
    for i in target:
        f = open(format(where_to_exstract)+"\\"+format(count)+'.txt', 'w')
        f.write(i)
        count=count+1
        f.close()

    train_dir_syn = dataset / 'train_extract' / 'time_based_blind'
    extract_syn = pd.read_csv(train_dir_syn / 'time_based.csv', engine ='python')
    where_to_exstract = dataset / 'train' / 'time_based_blind'
    target = extract_syn.pop('Info')
    count = 0
    for i in target:
        f = open(format(where_to_exstract) + "\\" + format(count) + '.txt', 'w')
        f.write(i)
        count = count + 1
        f.close()

    train_dir_udp = dataset / 'train_extract' / 'blacklist'
    extract_udp = pd.read_csv(train_dir_udp / 'blacklist.csv', engine ='python')
    where_to_exstract = dataset / 'train' / 'blacklist'
    target = extract_udp.pop('Info')
    count = 0
    for i in target:
        f = open(format(where_to_exstract) + "\\" + format(count) + '.txt', 'w')
        f.write(i)
        count = count + 1
        f.close()

    train_dir_harmless = dataset / 'train_extract' / 'double_query'
    extract_harmless = pd.read_csv(train_dir_harmless / 'double_query.csv', engine ='python')
    where_to_exstract = dataset / 'train' / 'double_query'
    target = extract_harmless.pop('Info')
    count = 0
    for i in target:
        f = open(format(where_to_exstract) + "\\" + format(count) + '.txt', 'w')
        f.write(i)
        count = count + 1
        f.close()




    ## now test vaulues

    train_dir_ack = dataset / 'test_extract' / 'error_based'
    extract_ack = pd.read_csv(train_dir_ack / 'test_error_based.csv', engine ='python')
    where_to_exstract = dataset / 'test' / 'error_based'
    target = extract_ack.pop('Info')
    count = 0
    for i in target:
        f = open(format(where_to_exstract) + "\\" + format(count) + '.txt', 'w')
        f.write(i)
        count = count + 1
        f.close()

    train_dir_http = dataset / 'test_extract' / 'boolean_based_blind'
    extract_http = pd.read_csv(train_dir_http / 'test_blind.csv', engine ='python')
    where_to_exstract = dataset / 'test' / 'boolean_based_blind'
    target = extract_http.pop('Info')
    count = 0
    for i in target:
        f = open(format(where_to_exstract) + "\\" + format(count) + '.txt', 'w')
        f.write(i)
        count = count + 1
        f.close()

    train_dir_syn = dataset / 'test_extract' / 'time_based_blind'
    extract_syn = pd.read_csv(train_dir_syn / 'test_time_based.csv', engine ='python')
    where_to_exstract = dataset / 'test' / 'time_based_blind'
    target = extract_syn.pop('Info')
    count = 0
    for i in target:
        f = open(format(where_to_exstract) + "\\" + format(count) + '.txt', 'w')
        f.write(i)
        count = count + 1
        f.close()

    train_dir_udp = dataset / 'test_extract' / 'blacklist'
    extract_udp = pd.read_csv(train_dir_udp / 'test_blacklist.csv', engine ='python')
    where_to_exstract = dataset / 'test' / 'blacklist'
    target = extract_udp.pop('Info')
    count = 0
    for i in target:
        f = open(format(where_to_exstract) + "\\" + format(count) + '.txt', 'w')
        f.write(i)
        count = count + 1
        f.close()

    train_dir_udp = dataset / 'test_extract' / 'double_query'
    extract_udp = pd.read_csv(train_dir_udp / 'double_query_test.csv', engine ='python')
    where_to_exstract = dataset / 'test' / 'double_query'
    target = extract_udp.pop('Info')
    count = 0
    for i in target:
        f = open(format(where_to_exstract) + "\\" + format(count) + '.txt', 'w')
        f.write(i)
        count = count + 1
        f.close()




def plot_metrics(history):
  metrics =  ['loss', 'accuracy', 'precision', 'recall']
  for n, metric in enumerate(metrics):
    name = metric.replace("_"," ").capitalize()
    plt.subplot(2,2,n+1)
    plt.plot(history.epoch,  history.history[metric], color=colors[0], label='Train')
    plt.plot(history.epoch, history.history['val_'+metric],
             color=colors[0], linestyle="--", label='Val')
    plt.xlabel('Epoch')
    plt.ylabel(name)
    if metric == 'loss':
      plt.ylim([0, plt.ylim()[1]])
    elif metric == 'accuracy':
      plt.ylim([0.8,1])
    else:
      plt.ylim([0,3])

    plt.legend()
  plt.show()







def int_vectorize_text(text, label):
  text = tf.expand_dims(text, -1)
  return int_vectorize_layer(text), label
#dataset=os.path.abspath(".keras")

def configure_dataset(dataset):
  return dataset.cache().prefetch(buffer_size=AUTOTUNE)

def create_model(vocab_size, num_labels):
  #model = tf.keras.Sequential([
  #    layers.Embedding(vocab_size, 64, mask_zero=True),
  #    layers.Conv1D(64, 5, padding="valid", activation="relu", strides=2),
  #    layers.GlobalMaxPooling1D(),
  #    layers.Dense(num_labels)
  #])
  model = tf.keras.Sequential()
  model.add(layers.Embedding(vocab_size, 64, input_length=MAX_SEQUENCE_LENGTH))
  model.add(layers.Conv1D(64, 5, strides=2, padding="valid", activation="relu"))#,kernel_regularizer=regularizers.l2(0.01)))
  model.add(layers.GlobalMaxPooling1D())
  model.add(layers.Flatten())
  model.add(layers.Dense(num_labels))
  return model


def get_string_labels(predicted_scores_batch):
  predicted_int_labels = tf.argmax(predicted_scores_batch, axis=1)
  predicted_labels = tf.gather(raw_train_ds.class_names, predicted_int_labels)
  return predicted_labels

##### функции предобработки

extract()


######


dataset=pathlib.Path(".keras")

train_dir = dataset/'train'
print(list(train_dir.iterdir()))

#sample_file = train_dir/'ack/1755.txt'
#with open(sample_file) as f:
#  print(f.read())

labels_batches=[]
labels_batches_train=[]
batch_size = 32
seed = 42

raw_train_ds = preprocessing.text_dataset_from_directory(
    train_dir,
    batch_size=batch_size,
    validation_split=0.2,
    subset='training',
    seed=seed)
#print(len(raw_train_ds))
for text_batch, label_batch in raw_train_ds.take(1):
  for i in range(32):#10):
    print("SQLi: ", text_batch.numpy()[i][:100], '...')
    print("Label:", label_batch.numpy()[i])
    labels_batches_train.append(label_batch.numpy()[i])

for i, label in enumerate(raw_train_ds.class_names):
  print("Label", i, "corresponds to", label)


raw_val_ds = preprocessing.text_dataset_from_directory(
    train_dir,
    batch_size=batch_size,
    validation_split=0.2,
    subset='validation',
    seed=seed)

test_dir = dataset/'test'
raw_test_ds = preprocessing.text_dataset_from_directory(
    test_dir, batch_size=batch_size)

for text_batch_test, label_batch_test in raw_test_ds.take(1):
  print("Test SQLis:")
  for i in range(32):#10):
    print("SQLi: ", text_batch_test.numpy()[i][:100], '...')
    print("Label:", label_batch_test.numpy()[i])
    labels_batches.append(label_batch_test.numpy()[i])



#VOCAB_SIZE = 5852#10000


#MAX_SEQUENCE_LENGTH = 250

int_vectorize_layer = TextVectorization(
    max_tokens=VOCAB_SIZE,
    output_mode='int',
    output_sequence_length=MAX_SEQUENCE_LENGTH)
#

#
# Make a text-only dataset (without labels), then call adapt
train_text = raw_train_ds.map(lambda text, labels: text)
int_vectorize_layer.adapt(train_text)

# Retrieve a batch (of 32 reviews and labels) from the dataset
text_batch, label_batch = next(iter(raw_train_ds))
first_question, first_label = text_batch[16], label_batch[16]
print("SQLi", first_question)
print("Label", first_label)

print("'int' vectorized SQLi:",
      int_vectorize_text(first_question, first_label)[0])

print("3 ---> ", int_vectorize_layer.get_vocabulary()[3])
print("457 ---> ", int_vectorize_layer.get_vocabulary()[457])
print("Vocabulary size: {}".format(len(int_vectorize_layer.get_vocabulary())))

int_train_ds = raw_train_ds.map(int_vectorize_text)
int_val_ds = raw_val_ds.map(int_vectorize_text)
int_test_ds = raw_test_ds.map(int_vectorize_text)

AUTOTUNE = tf.data.experimental.AUTOTUNE

int_train_ds = configure_dataset(int_train_ds)
int_val_ds = configure_dataset(int_val_ds)
int_test_ds = configure_dataset(int_test_ds)




print('\n')
for text_train_ds, label_train_ds in int_train_ds.take(1):
  for i in range(32):#10):
    print("SQLi: ", text_train_ds.numpy()[i][:100], '...')
    print("Label:", label_train_ds.numpy()[i])


print('\n')
for text_test_ds, label_test_ds in int_test_ds.take(1):
  for i in range(32):#10):
    print("SQLi: ", text_test_ds.numpy()[i][:100], '...')
    print("Label:", label_test_ds.numpy()[i])








# train model

# vocab_size is VOCAB_SIZE + 1 since 0 is used additionally for padding.
int_model = create_model(vocab_size=VOCAB_SIZE + 1, num_labels=5)
int_model.compile(
    loss=losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer='adam',
    metrics=['accuracy',recall,precision])#['accuracy'])
history = int_model.fit(int_train_ds, validation_data=int_val_ds, epochs=22)#5)

print("ConvNet model on int vectorized data:")
print(int_model.summary())


int_loss, int_accuracy,int_recall,int_precision = int_model.evaluate(int_test_ds)

print("Int model accuracy: {:2.2%}".format(int_accuracy))

#export model

export_model = tf.keras.Sequential(
    [int_vectorize_layer, int_model,
     layers.Activation('sigmoid')])

export_model.compile(
    loss=losses.SparseCategoricalCrossentropy(from_logits=False),
    optimizer='adam',
    metrics=['accuracy',recall,precision])

# Test it with `raw_test_ds`, which yields raw strings
loss, accuracy,recall1,precision1 = export_model.evaluate(raw_test_ds)
print("Accuracy: {:2.2%}".format(int_accuracy))



inputs = [
     "(select if( (select substr(email_id,9,1) from emails limit 0,1)='m',sleep(10),null))--+",   # time_based
     "(select count(*), concat(0x3a,0x3a,(select database()),0x3a,0x3a, floor(rand()*2))a from information_schema.tables group by a) --+",   # double query
]

predicted_scores = export_model.predict(inputs)
predicted_labels = get_string_labels(predicted_scores)
for input, label in zip(inputs, predicted_labels):
    print("SQLi: ", input)
    print("Predicted label: ", label.numpy())
    print('\n')


#int_model.save('my_model')


####graphuics

history_dict = history.history
print(history_dict.keys())


acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

#or
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

plt.show()


###### test-graphics



plot_metrics(history)

text_batch_test=text_batch_test.numpy()

y_pred=export_model.predict(text_batch_test)



list_label = []
counter=0
predicted_labels = get_string_labels(y_pred)
for label in  predicted_labels:
  if label.numpy()==b'blacklist':
      list_label.append(0)
  if label.numpy()==b'boolean_based_blind':
      list_label.append(1)
  if label.numpy()==b'double_query':
      list_label.append(2)
  if label.numpy()==b'error_based':
      list_label.append(3)
  if label.numpy()==b'time_based_blind':
      list_label.append(4)

list_label=np.array(list_label)

labels_batches=np.array(labels_batches)

confusion = confusion_matrix(labels_batches, list_label)
print('Confusion Matrix\n')
print(confusion)

print('\nAccuracy: {:.2f}\n'.format(accuracy_score(labels_batches, list_label)))

print('Micro Precision: {:.2f}'.format(precision_score(labels_batches, list_label, average='micro')))
print('Micro Recall: {:.2f}'.format(recall_score(labels_batches, list_label, average='micro')))
print('Micro F1-score: {:.2f}\n'.format(f1_score(labels_batches, list_label, average='micro')))

print('Macro Precision: {:.2f}'.format(precision_score(labels_batches, list_label, average='macro')))
print('Macro Recall: {:.2f}'.format(recall_score(labels_batches, list_label, average='macro')))
print('Macro F1-score: {:.2f}\n'.format(f1_score(labels_batches, list_label, average='macro')))

print('Weighted Precision: {:.2f}'.format(precision_score(labels_batches, list_label, average='weighted')))
print('Weighted Recall: {:.2f}'.format(recall_score(labels_batches, list_label, average='weighted')))
print('Weighted F1-score: {:.2f}'.format(f1_score(labels_batches, list_label, average='weighted')))

print('\nClassification Report\n')
print(classification_report(labels_batches, list_label, target_names=['Class 0', 'Class 1', 'Class 2','Class 3', 'Class 4']))
#print(classification_report(list_label, labels_batches, target_names=['Class 0', 'Class 1', 'Class 2','Class 3', 'Class 4']))


########################ROC
#y_test = np.arange(160)

#y_test = np.arange(160).reshape(32,5)

y_test=np.zeros((32,5), dtype=int)

for y_test_matrix, label in zip(y_test, labels_batches):
    y_test_matrix[label]=1






n_classes=5


fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
colors = cycle(['blue', 'red', 'green','yellow','purple'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=1.5, label='ROC curve of class {0} (area = {1:0.2f})' ''.format(i, roc_auc[i]))
plt.plot([0, 1], [0, 1], 'k-', lw=1.5)
plt.xlim([-0.05, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic for multi-class data')
plt.legend(loc="lower right")
plt.show()




########### text vectorization sample
#print("text vectorization sample")
# Retrieve a batch (of 32 reviews and labels) from the dataset
#text_batch, label_batch = next(iter(raw_train_ds))
#first_question, first_label = text_batch[0], label_batch[0]
#print("Packet", first_question)
#print("Label", first_label)

#print("'int' vectorized packet:",int_vectorize_text(first_question, first_label)[0])
#vectorized_text=int_vectorize_text(first_question, first_label)[0]
#for i in vectorized_text:
#    for j in i:
#        print(j,"->",int_vectorize_layer.get_vocabulary()[j])






