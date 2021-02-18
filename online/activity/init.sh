#/bin/bash

wget https://archive.ics.uci.edu/ml/machine-learning-databases/00427/Datasets_Healthy_Older_People.zip
unzip Datasets_Healthy_Older_People.zip
cd Datasets_Healthy_Older_People
mv S1_Dataset/README.txt README_S1.txt
cd S1_Dataset/README.txt README_S2.txt
cat S1_Dataset/* S2_Dataset/* > data.csv