import librosa
import librosa.display
import matplotlib.pyplot as plt
import argparse
import numpy as np
import os
import pickle
import sys

from sklearn import preprocessing 
from sklearn.model_selection import KFold
from sklearn.utils import shuffle

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import concatenate, Dense
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import Model, load_model

from datagenerator import MultiDataGenerator


def load_set_model(models_dir, model_id, fold, fine_tune):

  model = load_model(models_dir + model_id + "_" + str(fold) + ".h5")

  if (fine_tune):

    i = 0
    for layer in model.layers[:-5]:
      layer._name = model_id + "_" + str(i)
      layer.trainable = False
      i += 1
    for layer in model.layers[-5:]:
      layer.trainable = True
      layer._name = model_id + "_" + str(i)
      i += 1

  else:

    i = 0
    for layer in model.layers:
      layer._name = model_id + "_" + str(i)
      layer.trainable = False
      i += 1

  return model


def define_architecture(models_dir, models_list, fold, n_classes, fine_tune, early_fuse):

  output_layer = -1
  if (early_fuse):
    output_layer = -2

  fusion_list = []
  all_inputs = []

  for m in models_list:
    model = load_set_model(models_dir, m, fold, fine_tune)
    fusion_list.append(model.layers[output_layer].output)
    all_inputs.append(model.input)

  fusion = concatenate(fusion_list)
  fusion = Dense(n_classes, activation='softmax')(fusion)

  model = Model(inputs=all_inputs, outputs=fusion)

  return model


def parser_args(cmd_args):

  parser = argparse.ArgumentParser(sys.argv[0], description="",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("-f", "--files_dir", type=str, action="store", default="../files/GTZAN/",
    help="Path to files containing the extracted features and labels")
  parser.add_argument("-r", "--representation", type=str, action="store", default="melspec",
    help="Type of features/representation to use for experiments")
  parser.add_argument("-b", "--batch_size", type=int, action="store", default=16,
    help="Batch size to train the neural network")
  parser.add_argument("-m", "--models", type=str, action="store", default="MM",
    help="(Pretrained) models to fuse - using the aronymns available in the paper",
    choices=["MM", "MMC", "MMCC", "MMCT", "MMCCT"])
  parser.add_argument("-t", "--fine_tuning", type=int, action="store", default=1,
    help="Whether the models will be fine-tuned or not (boolean)",
    choices=[0,1])  
  parser.add_argument("-e", "--early_fusion", type=int, action="store", default=1,
    help="Whether the models will be early-fused or not (boolean)",
    choices=[0,1])


  return parser.parse_args(cmd_args)

# obtaining arguments from command line
args = parser_args(sys.argv[1:])

files_dir = args.files_dir
if (files_dir[-1] != "/"):
  files_dir += "/"

batch_size = args.batch_size
fine_tune = args.fine_tuning
early_fuse = args.early_fusion

if not (os.path.exists(files_dir+"models/")):
    os.mkdir(files_dir+"models/")

switcher={
  "MM" : ["melspec","melfcc"],
  "MMC" : ["melspec","melfcc", "cqt"],
  "MMCC" : ["melspec","melfcc", "cqt", "chroma"],
  "MMCT" : ["melspec","melfcc", "cqt", "tempog"],
  "MMCCT" : ["melspec","melfcc", "cqt", "chroma", "tempog"]
}
feat_list = switcher.get(args.models)


# loading and encoding labels
f = open(files_dir+"labels.pkl","rb")
all_labels = pickle.load(f)
f.close()

label_encoder = preprocessing.LabelEncoder() 
encoded_labels = label_encoder.fit_transform(all_labels)

# some definitions for training the NNs
earlyStopping = EarlyStopping(monitor="val_loss", patience=20,
  verbose=0, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau()

n_examples = len(encoded_labels)
n_classes = len(np.unique(encoded_labels))

params = {"batch_size": batch_size,
          "n_classes": n_classes,
          "n_channels": 1,
          "gen_dir": files_dir+ "/",
          "feat_list": feat_list,
          "shuffle": True}


kf = KFold(n_splits=5, shuffle=True, random_state=42)
val_kf = KFold(n_splits=2, shuffle=True, random_state=42)

acc = []
k = 0
for train_index, test_index in kf.split(range(n_examples)):
    
  this_tr, this_val = next(val_kf.split(train_index))

  # Generators
  training_generator = MultiDataGenerator(train_index[this_tr],
    encoded_labels,
    **params)
  val_generator = MultiDataGenerator(train_index[this_val],
    encoded_labels,
    **params)
  test_generator = MultiDataGenerator(test_index,
    encoded_labels,
    **params)

  model = define_architecture(models_dir=files_dir+"models/", 
    models_list=feat_list, fold=k, n_classes=n_classes, 
    fine_tune=fine_tune, early_fuse=early_fuse)
  model.compile(optimizer=tf.keras.optimizers.Adam(),
    loss="categorical_crossentropy",
    metrics=["accuracy"])

  model.fit(x = training_generator,
    validation_data = val_generator,
    callbacks = [earlyStopping, reduce_lr],
    epochs = 3)

  score, this_acc = model.evaluate(x=test_generator)
  acc.append(this_acc)

  k += 1

print(np.mean(acc))
print(np.std(acc))
