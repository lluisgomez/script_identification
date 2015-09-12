
## Compile

```
cmake .
make
```

## Use:


First thing to do is to create a folder "data" and unzip there the training set, then create a folder "data/LearnedFilters". Then create a subfolder for each Script class, e.g.:
```
for i in `ls data/TrainDataset_CVSI2015/`; do echo $i; mkdir data/LearnedFilters/$i; done
```

1. Learn the filter banks and save them into "data/LearnedFilters/":
```
./extract_filters data/TrainDataset_CVSI2015/Arabic_labels.txt data/TrainDataset_CVSI2015/Bengali_labels.txt data/TrainDataset_CVSI2015/English_labels.txt data/TrainDataset_CVSI2015/Gujrathi_labels.txt data/TrainDataset_CVSI2015/Hindi_labels.txt data/TrainDataset_CVSI2015/Kannada_labels.txt data/TrainDataset_CVSI2015/Oriya_labels.txt data/TrainDataset_CVSI2015/Punjabi_labels.txt data/TrainDataset_CVSI2015/Tamil_labels.txt data/TrainDataset_CVSI2015/Telegu_labels.txt
```

2. Extract Training Features:
```
for i in `ls data/TrainDataset_CVSI2015/*txt | cut -d "_" -f 1,2`; do echo $i; ./extract_features first_layer_centroids.xml $i"_labels.txt" > $i"_features.csv" ; done
```
