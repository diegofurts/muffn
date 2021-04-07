# Presentation
This is a repo to provide reproducibility to the results presented in the paper "On the Fusion of Multiple Audio Representations for Music Genre Classification," submitted to the Multi-view Representation Learning and Multi-modal Information Representation special issue of Pattern Recognition Letters. Here, you can find the codes for all the three steps: feature generation, individual classifier training/evaluation, and MUFFN (Multi-Feature Fusion Networks) training/evaluation.

If the paper gets accepted, a better version of the codes will be available at the first authors' repository. Specifically, we are modifying the code to be more flexible to users that want to use MUFFN with alternative features or for different tasks.

# How to use it
## Required packages
We used the following packages to run our experiments:
```
tensorflow 2.2.0
librosa 0.6.3
sklearn 0.22.2
numpy 1.18.4
```

## Python files
To reproduce our experiments, you'll need to run three files in order. First, execute ```extractfeatures.py```. This is necessary so we extract the features only once and reuse that in all experiments.

Next, you need to train the inidividaul models, by runing ```individual.py```. By default, it will run an experiment on GTZAN using melspectrogram (note that we make available a very small subset of GTZAN for sanity check and demonstration). For custom training, use the following options:
```
-f FILES_DIR, --files_dir FILES_DIR
                      Path to files containing the extracted features and
                      labels (default: ../files/GTZAN/)
-r {chroma,cqt,harms,melfcc,melspec,ssm,tempog,tonnz}, --representation {chroma,cqt,harms,melfcc,melspec,ssm,tempog,tonnz}
                      Type of features/representation to use for experiments
                      (default: melspec)
-b BATCH_SIZE, --batch_size BATCH_SIZE
                      Batch size to train the neural network (default: 16)
```

Finally, you can run the experiments on MUFFN using ```muffn.py```. For this, use the following options:
```
-f FILES_DIR, --files_dir FILES_DIR
                    Path to files containing the extracted features and
                    labels (default: ../files/GTZAN/)
-r REPRESENTATION, --representation REPRESENTATION
                    Type of features/representation to use for experiments
                    (default: melspec)
-b BATCH_SIZE, --batch_size BATCH_SIZE
                    Batch size to train the neural network (default: 16)
-m {MM,MMC,MMCC,MMCT,MMCCT}, --models {MM,MMC,MMCC,MMCT,MMCCT}
                    (Pretrained) models to fuse - using the aronymns
                    available in the paper (default: MM)
-t {0,1}, --fine_tuning {0,1}
                    Whether the models will be fine-tuned or not (boolean)
                    (default: 1)
-e {0,1}, --early_fusion {0,1}
                    Whether the models will be early-fused or not
                    (boolean) (default: 1)
```
