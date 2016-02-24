
# Scene Text Script Identification with Ensembles of conjoined Networks

Script Identification using an Ensemble of Conjoined Networks (ECN) and a loss function that takes into account the global classification error for a group of N patches instead of looking only into a single image patch. 

At training time our network is presented with a group of N patches sharing the same class label and produces a single probability distribution over the classes for all them.

## Compilation

Dependencies: Caffe must be installed in the host system. The python scripts to extract image patches depend on numpy, scipy, and skimage. 


```
cd tools/
cmake .
make
```

## Demo

Reproduce the results on the SIW13 dataset:


# Prepare the data

First of all you must copy the SIW13 original file stricture into data/orig/ folder. Then we densely extract patches from the original images and build a txt file with "patch-filename label" format :

```
cd data/

python ../scripts/extract_test_patches.py > ../caffe_models/siw13_simple_5_3_3_1/test_data.txt
python ../scripts/extract_train_patches.py > ../caffe_models/siw13_simple_5_3_3_1/train_data.txt
```

Then we do the same but in this case for groups of 10 patches (this is the input to the ECN) :

```
python ../scripts/extract_train_patches_ECN.py > ../../siw13_ECNx10_5_3_3_1/train_data.txt
```

Now we build the train/val databases and the mean train image in order to train the simple network first:

```
cd ../caffe_models/siw13_simple_5_3_3_1/

/path_to_caffe/build/tools/convert_imageset -backend leveldb -gray -shuffle ./ train_data.txt train_leveldb
/path_to_caffe/build/tools/convert_imageset -backend leveldb -gray -shuffle ./ test_data.txt test_leveldb

/path_to_caffe/build/tools/compute_image_mean -backend leveldb ./train_leveldb image_mean.binaryproto
```

Same for the ECN:

```
cd ../siw13_ECNx10_5_3_3_1/

../../tools/convert_imageset_ECN -backend leveldb -gray -shuffle ./ test_data.txt test_leveldb

../../tools/compute_image_mean_ECN -backend leveldb ../siw13_simple_5_3_3_1/train_leveldb image_mean.binaryproto 

```

# Training

First we train the simple network:


```
cd ../

/path_to_caffe/build/tools/caffe train --solver=siw13_simple_5_3_3_1/solver.prototxt
```

Then we do the finetuning on the ECN:


```
/path_to_caffe/build/tools/caffe train --solver=siw13_ECNx10_5_3_3_1/solver.prototxt --weights=siw13_simple_5_3_3_1/siw13_simple_5_3_3_1_train_iter_110000.caffemodel
```

# Test

Now you have the ECN trained model, and can deploy it on test data to obtain SIW13 overall classification accuracy as follows:

```
./tools/eval_siw13 caffe_models/siw13_simple_5_3_3_1/deploy.prototxt caffe_models/siw13_ECNx10_5_3_3_1/siw13_ECNx10_5_3_3_1_train_iter_50000.caffemodel caffe_models/siw13_simple_5_3_3_1/image_mean.binaryproto caffe_models/siw13_simple_5_3_3_1/labels.txt caffe_models/siw13_simple_5_3_3_1/test_data.txt
```

## Pre-trained

If you want to use our pre-trained models you can download them from:

 - https://www.dropbox.com/s/ikro6hh7jp199qr/siw13_simple_5_3_3_1_train_iter_450000.caffemodel?dl=0
 - https://www.dropbox.com/s/swactd7g8o0zl7x/siw13_ECNx10_5_3_3_1_train_iter_50000.caffemodel?dl=0

data files must be placed in the same folder as their respective prototxt files. Then you can evaluate on the test set as explained before.


