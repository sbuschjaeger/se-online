#/bin/bash

wget https://archive.ics.uci.edu/ml/machine-learning-databases/00224/Dataset.zip
unzip Dataset.zip
sed -i 's/[0-9]*\://'g Dataset/batch1.dat
sed -i 's/[0-9]*\://'g Dataset/batch2.dat
sed -i 's/[0-9]*\://'g Dataset/batch3.dat
sed -i 's/[0-9]*\://'g Dataset/batch4.dat
sed -i 's/[0-9]*\://'g Dataset/batch5.dat
sed -i 's/[0-9]*\://'g Dataset/batch6.dat
sed -i 's/[0-9]*\://'g Dataset/batch7.dat
sed -i 's/[0-9]*\://'g Dataset/batch8.dat
sed -i 's/[0-9]*\://'g Dataset/batch9.dat
sed -i 's/[0-9]*\://'g Dataset/batch10.dat