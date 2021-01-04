#!/bin/bash

# install conda
# latest miniconda (has issues with bootstrap) -> https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
#wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py38_4.8.3-Linux-x86_64.sh -O ~/miniconda.sh \
#    && /bin/bash ~/miniconda.sh -b -p $HOME/conda
#
#echo -e '\nexport PATH=$HOME/conda/bin:$PATH' >> $HOME/.bashrc && source $HOME/.bashrc

sudo python3 -m pip install pandas==1.0.5
sudo python3 -m pip install pyarrow==0.13.0
sudo python3 -m pip install statsmodels==0.10.0rc2 --pre
sudo python3 -m pip install fsspec s3fs