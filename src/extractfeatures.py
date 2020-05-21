import argparse
import click
import librosa
import numpy as np
import os
import pickle
import sys
import time


def parser_args(cmd_args):

    parser = argparse.ArgumentParser(sys.argv[0], description="Feature extraction for MUFFN experiments",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d", '--dataset_dir', type=str, action="store", default='../data/GTZAN/',
                        help="Path to the dataset's root folder")
    parser.add_argument("-f", '--files_dir', type=str, action="store", default='../files/GTZAN/',
                        help="Path to files to store the extracted features and labels")
    parser.add_argument("-s", '--duration', type=int, action="store", default=30,
                        help="Duration (in seconds) of each audio")

    return parser.parse_args(cmd_args)


# obtaining arguments from command line
print(sys.argv[1:])
args = parser_args(sys.argv[1:])

data_dir = args.dataset_dir
if (data_dir[-1] != '/'):
  data_dir += '/'

files_dir = args.files_dir
if (files_dir[-1] != '/'):
  files_dir += '/'

duration = args.duration

print('Welcome :]')
print('Your features will be extracted')

# to create the label vector
# we are assuming each genre is
# in a separate folder
files = []
labels = os.listdir(data_dir)

for l in labels:
    if os.path.isdir(data_dir+l):
        for f in os.listdir(data_dir+l):
            files.append({'label' : l, 'file' : (data_dir+l+'/'+f)})

print('files list: done')

# creating a single folder for each feature
if not (os.path.exists(files_dir)):
    os.mkdir(files_dir)
if not (os.path.exists(files_dir + 'melspec/')):
    os.mkdir(files_dir + 'melspec/')
if not (os.path.exists(files_dir + 'melfcc/')):
    os.mkdir(files_dir + 'melfcc/')
if not (os.path.exists(files_dir + 'chroma/')):
    os.mkdir(files_dir + 'chroma/')
if not (os.path.exists(files_dir + 'tempog/')):
    os.mkdir(files_dir + 'tempog/')
if not (os.path.exists(files_dir + 'tonnz/')):
    os.mkdir(files_dir + 'tonnz/')
if not (os.path.exists(files_dir + 'ssm/')):
    os.mkdir(files_dir + 'ssm/')
if not (os.path.exists(files_dir + 'harms/')):
    os.mkdir(files_dir + 'harms/')
if not (os.path.exists(files_dir + 'cqt/')):
    os.mkdir(files_dir + 'cqt/')


# finally, the feature extraction
all_labels = []
i = 0
with click.progressbar(files, fill_char='=', empty_char=' ') as bar:
    for f in bar:

        # print(str(i))

        y,sr = librosa.load(f['file'])
        if (len(y) < sr*duration):
            to_add = np.zeros((sr*duration)-len(y))
            y = np.hstack((y,to_add))
        elif (len(y) > sr*duration):
            y = y[:sr*duration]

        # melspectrogram
        melspec = librosa.feature.melspectrogram(y,sr,hop_length=1024)
        melspec[melspec==0] = np.finfo(float).eps
        melspec = np.log(melspec)
        np.save(files_dir+'melspec/'+str(i)+".npy", melspec)

        # chromagram
        chroma = librosa.feature.chroma_cqt(y,sr,hop_length=1024)
        np.save(files_dir+'chroma/'+str(i)+".npy", chroma)

        # tempogram
        tempog = librosa.feature.tempogram(y,sr,hop_length=1024,win_length=192)
        np.save(files_dir+'tempog/'+str(i)+".npy", tempog)

        # mfcc
        melfcc = librosa.feature.mfcc(y,sr,hop_length=1024)
        np.save(files_dir+'melfcc/'+str(i)+".npy", melfcc)

        # tonnetz
        tonnz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
        np.save(files_dir+'tonnz/'+str(i)+".npy", tonnz)

        # ssm
        chroma = librosa.feature.chroma_cens(y=y, sr=sr, hop_length=2048, win_len_smooth=2)
        ssm = librosa.segment.recurrence_matrix(chroma, mode='affinity')
        np.save(files_dir+'ssm/'+str(i)+".npy", ssm)

        # interp_hatmonics
        tempi = np.mean(librosa.feature.tempogram(y, sr), axis=1)
        h_range = [1, 2, 3, 4, 5, 6, 7, 8]
        f_tempo = librosa.tempo_frequencies(len(tempi), sr=sr)
        harms = librosa.interp_harmonics(tempi, f_tempo, h_range)
        np.save(files_dir+'harms/'+str(i)+".npy", harms)

        # cqt
        cqt = np.abs(librosa.cqt(y, sr, hop_length=1024))
        np.save(files_dir+'cqt/'+str(i)+".npy", cqt)

        all_labels.append(f["label"])
        i += 1

f = open(files_dir+"labels.pkl","wb")
pickle.dump(all_labels,f)
f.close()

print('feature extraction: done')