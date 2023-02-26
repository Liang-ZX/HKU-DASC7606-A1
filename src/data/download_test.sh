#! /bin/bash

############### Download Test Images ###############
# Google Drive sharing link
fileid='1-I1Rp1VrF4S5hx9_X3MUL50C10Ph1Tqe'

# Download zip dataset from Google Drive
filename='test.zip'

wget --wait 10 --random-wait --continue --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=${fileid}' -O- | sed -rn 's/.confirm=([0-9A-Za-z_]+)./\1\n/p')&id=${fileid}" -O ${filename} && rm -rf /tmp/cookies.txt

# Unzip
unzip ${filename}
rm ${filename}

mv "test" "ass1_dataset/"
