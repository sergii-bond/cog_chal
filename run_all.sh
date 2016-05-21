#!/bin/sh

echo "--Download modified images from Amazon AWS and extract them"
wget https://s3-us-west-1.amazonaws.com/cogniac-public-data/python_coding_challenge.tar.gz
tar zxvf python_coding_challenge.tar.gz
echo "Done----------------------"

echo "--Download images from flickr"
mkdir flickr_photos
python get_flickr.py
echo "Done----------------------"

echo "--Extract features from modified photos"
th extractor.lua -im_dir modified_images -out_file modified_features-101.t7
echo "Done----------------------"

echo "--Extract features from original photos"
th extractor.lua -im_dir flickr_photos -out_file flickr_features-101.t7
echo "Done----------------------"

echo "--Calculate similarity"
th similarity.lua
echo "Done----------------------"

echo "--Map predictions to original urls"
python map.py
echo "Done----------------------"

echo "Check accuracy on a validation set"
python score.py pred.csv
echo "Done----------------------"


