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
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Input, Conv2D, Dense, concatenate
from tensorflow.keras.layers import BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D
from tensorflow.keras.models import Model, load_model


from datagenerator import DataGenerator


def define_architecture(inputsize):

  in_x1 = Input(shape=inputsize)
  x1 = Conv2D(32, (3, 3), activation="relu", padding="same")(in_x1)
  x1 = BatchNormalization()(x1)
  x1 = MaxPooling2D((1, 4), padding="same", name="x1")(x1)

  inception_a1 = BatchNormalization()(x1)
  inception_a1 = Conv2D(32, (1, 1), activation="relu", name="inception_a1")(inception_a1)

  inception_a2 = BatchNormalization()(x1)
  inception_a2 = Conv2D(32, (1, 1), activation="relu")(inception_a2)
  inception_a2 = BatchNormalization()(inception_a2)
  inception_a2 = Conv2D(32, (3, 3), activation="relu", padding="same", name="inception_a2")(inception_a2)

  inception_a3 = BatchNormalization()(x1)
  inception_a3 = Conv2D(32, (1, 1), activation="relu")(inception_a3)
  inception_a3 = BatchNormalization()(inception_a3)
  inception_a3 = Conv2D(32, (5, 5), activation="relu", padding="same", name="inception_a3")(inception_a3)

  inception_a4 = MaxPooling2D((3, 3), padding="same", strides=1)(x1)
  inception_a4 = BatchNormalization()(inception_a4)
  inception_a4 = Conv2D(32, (1, 1), activation="relu", name="inception_a4")(inception_a4)

  # -----

  x2 = concatenate([x1, inception_a1, inception_a2,
    inception_a3, inception_a4])

  inception_b1 = BatchNormalization()(x2)
  inception_b1 = Conv2D(32, (1, 1), activation="relu", name="inception_b1")(inception_b1)

  inception_b2 = BatchNormalization()(x2)
  inception_b2 = Conv2D(32, (1, 1), activation="relu")(inception_b2)
  inception_b2 = BatchNormalization()(inception_b2)
  inception_b2 = Conv2D(32, (3, 3), activation="relu", padding="same", name="inception_b2")(inception_b2)

  inception_b3 = BatchNormalization()(x2)
  inception_b3 = Conv2D(32, (1, 1), activation="relu")(inception_b3)
  inception_b3 = BatchNormalization()(inception_b3)
  inception_b3 = Conv2D(32, (5, 5), activation="relu", padding="same", name="inception_b3")(inception_b3)

  inception_b4 = MaxPooling2D((3, 3), padding="same", strides=1)(x2)
  inception_b4 = BatchNormalization()(inception_b4)
  inception_b4 = Conv2D(32, (1, 1), activation="relu", name="inception_b4")(inception_b4)

  # -----

  x3 = concatenate([x2, inception_b1, inception_b2,
    inception_b3, inception_b4])

  inception_c1 = BatchNormalization()(x3)
  inception_c1 = Conv2D(32, (1, 1), activation="relu", name="inception_c1")(inception_c1)

  inception_c2 = BatchNormalization()(x3)
  inception_c2 = Conv2D(32, (1, 1), activation="relu")(inception_c2)
  inception_c2 = BatchNormalization()(inception_c2)
  inception_c2 = Conv2D(32, (3, 3), activation="relu", padding="same", name="inception_c2")(inception_c2)

  inception_c3 = BatchNormalization()(x3)
  inception_c3 = Conv2D(32, (1, 1), activation="relu")(inception_c3)
  inception_c3 = BatchNormalization()(inception_c3)
  inception_c3 = Conv2D(32, (5, 5), activation="relu", padding="same", name="inception_c3")(inception_c3)

  inception_c4 = MaxPooling2D((3, 3), padding="same", strides=1)(x3)
  inception_c4 = BatchNormalization()(inception_c4)
  inception_c4 = Conv2D(32, (1, 1), activation="relu", name="inception_c4")(inception_c4)

  # -----

  x4 = concatenate([x3, inception_c1, inception_c2,
    inception_c3, inception_c4])
  x4 = BatchNormalization()(x4)
  x4 = Conv2D(32, (1, 1), activation="relu")(x4)
  x4 = AveragePooling2D((2, 2), strides=2)(x4)
  x4 = BatchNormalization()(x4)
  x4 = GlobalAveragePooling2D(name="features")(x4)
  x4 = Dense(10, activation="softmax", name="dense")(x4)

  # -----

  return Model(in_x1,x4)

def parser_args(cmd_args):

  parser = argparse.ArgumentParser(sys.argv[0], description="",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("-f", "--files_dir", type=str, action="store", default="../files/GTZAN/",
    help="Path to files containing the extracted features and labels")
  parser.add_argument("-r", "--representation", type=str, action="store", default="melspec",
    help="Type of features/representation to use for experiments",
    choices=["chroma", "cqt", "harms", "melfcc", "melspec", "ssm", "tempog", "tonnz"])
  parser.add_argument("-b", "--batch_size", type=int, action="store", default=16,
    help="Batch size to train the neural network")

  return parser.parse_args(cmd_args)

# obtaining arguments from command line
args = parser_args(sys.argv[1:])

files_dir = args.files_dir
if (files_dir[-1] != "/"):
  files_dir += "/"

batch_size = args.batch_size
repr_id = args.representation

if not (os.path.exists(files_dir+"models/")):
    os.mkdir(files_dir+"models/")

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

example_data = np.load(files_dir + repr_id + "/0.npy")
input_layer = (example_data.shape[0], example_data.shape[1], 1)
n_examples = len(encoded_labels)
n_classes = len(np.unique(encoded_labels))

params = {"dim": (example_data.shape[0], example_data.shape[1]),
          "batch_size": batch_size,
          "n_classes": n_classes,
          "n_channels": 1,
          "gen_dir": files_dir+ repr_id + "/",
          "shuffle": True}


kf = KFold(n_splits=5, shuffle=True, random_state=42)
val_kf = KFold(n_splits=10, shuffle=True, random_state=42)

acc = []
k = 0
for train_index, test_index in kf.split(range(n_examples)):
    
  # if (os.path.exists(files_dir+"models/"+repr_id+"_"+str(k)+".h5")):
  #     model = load_model(files_dir+"models/"+repr_id+"_"+str(k)+".h5")
      
  #     test_generator = DataGenerator(test_index,
  #                                encoded_labels,
  #                                **params)

  #     score, this_acc = model.evaluate(x=test_generator)
  #     acc.append(this_acc)
  #     k+=1
  #     continue

  this_tr, this_val = next(val_kf.split(train_index))

  # Generators
  training_generator = DataGenerator(train_index[this_tr],
    encoded_labels,
    **params)
  val_generator = DataGenerator(train_index[this_val],
    encoded_labels,
    **params)
  test_generator = DataGenerator(test_index,
    encoded_labels,
    **params)

  model = define_architecture(input_layer)
  model.compile(optimizer=tf.keras.optimizers.Adam(),
    loss="categorical_crossentropy",
    metrics=["accuracy"])

  model.fit(x = training_generator,
    validation_data = val_generator,
    callbacks = [earlyStopping, reduce_lr],
    epochs = 50)

  score, this_acc = model.evaluate(x=test_generator)
  acc.append(this_acc)

  model.save(files_dir+"models/"+repr_id+"_"+str(k)+".h5")

  k += 1

print(np.mean(acc))
print(np.std(acc))