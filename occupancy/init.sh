#/bin/bash

wget https://archive.ics.uci.edu/ml/machine-learning-databases/00357/occupancy_data.zip
unzip occupancy_data.zip
cat datatest.txt <(tail -n+2 datatraining.txt) <(tail -n+2 datatest2.txt) > data.csv