#!/bin/bash

data_directory="data"
train_directory="${data_directory}/train"

# download the data if necessary
if [ -d "$train_directory" ]; then
  echo "Data already downloaded."
# extract the data
else
  echo "Directory does not exists, downloading the data..."
  cd "$data_directory"
  kaggle competitions download -c iapr24-coin-counter
  unzip iapr24-coin-counter.zip && rm iapr24-coin-counter.zip
fi

# flatten the train data
cd "$train_directory"
find . -mindepth 2 -type f -name "*.JPG" -exec mv {} . \;
find . -mindepth 1 -type d -empty -delete
