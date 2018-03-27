## Installation
1. Install anaconda with Python 3.6
2. Run `conda env create -f environment.yaml` to create a new conda environment with the required dependencies and then activate the environment with `source activate understandable-faster-rcnn`
3. Install pytorch from source https://github.com/pytorch/pytorch#from-source with CUDA 9.1 (so use magma-cuda91) and then install torchvision with `pip install torchvision`