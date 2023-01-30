#! /bin/bash

# Google Drive sharing link
fileid='1WhC8AsloaEUipGCQQncYir9Q-Kb9meTC'

# Download zip dataset from Google Drive
filename='ass1_dataset.zip'

wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=${fileid}' -O- | sed -rn 's/.confirm=([0-9A-Za-z_]+)./\1\n/p')&id=${fileid}" -O ${filename} && rm -rf /tmp/cookies.txt

# Unzip
unzip ${filename}
rm ${filename}

mv "deeplearning_assignment_1_dataset" "ass1_dataset"
