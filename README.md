
# Scene Text Script Identification

Single Layer Convolutional Neural Net trained for script identification combined with a classifier based in Naive Bayes Nearest Neighbor (NBNN). For the NBNN classification we learn specific Image2Class distances by leveraging the search space topology.

## Compilation

Dependencies: OpenCV-3.0.0 must be installed in the host system. 

```
cmake .
make
```

## Demo

Reproduce the results on the CVSI2015 dataset:

```
./cvsi_ClassifyAllScripts /path/to/TestDataset_CVSI2015/*jpg OutCVSIResults.txt
```

Reproduce the results on the Babel_01 dataset:

```
./babel_ClassifyAllScripts /path/to/Babel_01/test/*jpg OutBabelResults.txt
```

will produce a single file with results for all the images (one per line).


## Data

Since our method uses Nearest Neighbor for classification all training data must be available to run our programs. All needed data files are compressed into a single bz2 file (929Mb) that can be downloaded here:

 - http://158.109.8.43/textlocation/script_identification_cvsi_data.tar.bz2
 - http://158.109.8.43/textlocation/script_identification_babel_data.tar.bz2

All data files must be placed in the same folder as the binaries.

## Train

To generate the convolutional kernels and training data see README.md file in train folder.

