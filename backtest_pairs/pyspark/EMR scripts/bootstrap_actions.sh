#!/usr/bin/env bash

# install conda
wget --quiet https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh \
    && /bin/bash ~/miniconda.sh -b -p $HOME/conda

echo -e '\nexport PATH=$HOME/conda/bin:$PATH' >> $HOME/.bashrc && source $HOME/.bashrc

# install packages
sudo pip3 install -U matplotlib
sudo pip3 install -U pandas
sudo pip3 install -U pyarrow==0.13.0
sudo pip3 install -U statsmodels==0.10.0rc2 --pre
sudo pip3 install -U seaborn
sudo pip3 install -U fsspec
sudo pip3 install -U s3fs


