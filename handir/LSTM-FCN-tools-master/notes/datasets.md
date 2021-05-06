# Datasets

## Use in LSTM-FCN-trainer

Used datasets are required to be listed in `utils/constants.py`. Here you need to list the location of the TRAIN dataset, TEST, how long the data is (how many datapoints on the "x" axis), the number of classes, and X/Y axis labels.

For example, the main datasets I've been working with as part of my research are `Seawater_all_TRAIN` (11 classes), `Seawater_all4cats_TRAIN` (4 classes), `Seawater2_TRAIN` (25 classes), `Explosives_TRAIN` (11 classes), and `Explosives_3cats_TRAIN` (3 classes). Each with their own corresponding TEST location. Seawater are 1002 in length while Explosives are 1502 in length.


## Where to find more data, if you need some

[UEA & UCR Time Series Classification Repository](http://www.timeseriesclassification.com/index.php) is the best resource for finding more curated time series data.

You can go to examples, like ArrowHead [here](http://www.timeseriesclassification.com/description.php?Dataset=ArrowHead) where you'll be given the option to download the dataset. After doing that, you'll have a .zip file (`ArrowHead.zip`). After unzipping, you'll see:

 - -rwxr-xr-x@   1 snd   142K Oct 18 10:03 ArrowHead_TRAIN.txt
 - -rwxr-xr-x@   1 snd   104K Oct 18 10:03 ArrowHead_TRAIN.arff
 - -rwxr-xr-x@   1 snd   689K Oct 18 10:03 ArrowHead_TEST.txt
 - -rwxr-xr-x@   1 snd   480K Oct 18 10:03 ArrowHead_TEST.arff
 - -rwxr-xr-x@   1 snd   586B Oct 18 10:03 ArrowHead.txt

Where the data looks like this (in `read.py`):

```
import pandas as pd


df = pd.read_csv('ArrowHead_TRAIN.txt', sep='\t', header=None)
print(df.head())
```

```
                                                   0
0     0.0000000e+00  -1.9630089e+00  -1.9578249e+...
1     1.0000000e+00  -1.7745713e+00  -1.7740359e+...
2     2.0000000e+00  -1.8660211e+00  -1.8419912e+...
3     0.0000000e+00  -2.0737575e+00  -2.0733013e+...
4     1.0000000e+00  -1.7462554e+00  -1.7412629e+...

```

If you look at `ArrowHead.txt`, they say "The three classes are called "Avonlea", "Clovis" and "Mix"." which is what the numeric label in the first column corresponds to.





