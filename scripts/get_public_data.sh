#!/bin/bash
# Script to download IDRBT Cheque Image Dataset
# Note: This is a placeholder as the actual dataset requires filling a form.
# We will create the directory structure and add a placeholder README.

mkdir -p ../data/idrbt_cheques
echo "Please download the IDRBT dataset from https://www.idrbt.ac.in/idrbt-cheque-image-dataset/ and place the images in data/idrbt_cheques/" > ../data/idrbt_cheques/README.txt

# If we had a direct link (which usually isn't public without form), we would wget it here.
# For this demo, we might want to generate some dummy checks if the user doesn't have the dataset.
