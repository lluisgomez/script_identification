
## Compile

```
cmake .
make
```

## Use:


1. Learn the filter banks and save them into "first_layer_filters.xml":
```
./extract_filters data/TrainDataset_CVSI2015/Arabic_labels.txt data/TrainDataset_CVSI2015/Bengali_labels.txt data/TrainDataset_CVSI2015/English_labels.txt data/TrainDataset_CVSI2015/Gujrathi_labels.txt data/TrainDataset_CVSI2015/Hindi_labels.txt data/TrainDataset_CVSI2015/Kannada_labels.txt data/TrainDataset_CVSI2015/Oriya_labels.txt data/TrainDataset_CVSI2015/Punjabi_labels.txt data/TrainDataset_CVSI2015/Tamil_labels.txt data/TrainDataset_CVSI2015/Telegu_labels.txt
```
where the txt files contain "filename label" for every image in the train dataset.

2. Extract Training Features:
```
for i in `ls data/TrainDataset_CVSI2015/*txt | cut -d "_" -f 1,2`; do echo $i; ./extract_features first_layer_filters.xml $i"_labels.txt" > $i"_features.csv" ; done
```
