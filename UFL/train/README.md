
## Compile

```
cmake .
make
```

## Use:


1. Learn the filter banks and save them into "first_layer_filters.xml":
```
./extract_filters data/TrainDataset_CVSI2015/*_labels.txt 
```
where the txt files contain "filename label" for every image in the train dataset.

2. Extract Training Features:
```
for i in `ls data/TrainDataset_CVSI2015/*_labels.txt | cut -d "_" -f 1,2`; do echo $i; ./extract_features first_layer_filters.xml $i"_labels.txt" > $i"_features.csv" ; done
```
