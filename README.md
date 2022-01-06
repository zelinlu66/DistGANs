# Distributed Generative Adversarial Neural Networks (DistGANs)

DistGANs is a Python package to perform distributed training of conditional generative adversarial neural networks for multi-class labeled image data.\
DistGANs partitionins the training data according to data labels, and enhances scalability by performing a parallel training where multiple generators are concurrently trained, each one of them focusing on a single data label.

This is a code implemented in collaboration with:

- Massimiliano Lupo Pasini at Oak Ridge National Laboratory (lupopasinim@ornl.gov)
- Nouamane Laanait at Oak Ridge National Laboratory (laanaitn@ornl.gov; nlaanait@gmail.com)
- Vittorio Gabbi at Politecnico di Milano (vittorio.gabbi@mail.polimi.it)
- Debangshu Mukherjee at Oak Ridge National Laboratory (mukherjeed@ornl.gov)
- Vitaliy Starchenko at Oak Ridge National Laboratory (starchenkov@ornl.gov)
- Junqi Yin at Oak Ridge National Laboratory (yinj@ornl.gov)
- Andrey Prokpenko at Oak Ridge National Laboratory (prokopenkoav@ornl.gov)

## Requirements
Python 3.5 or greater\
PyTorch (any version works)
Optional, if NVIDIA gpu is present:
```
pip install pycuda
```

## Code style

To keep similar code style, it should be formatted using [black](https://github.com/psf/black):

```
black -S -l 79 {source_file_or_directory}
```

## Quick start conda setup
```
conda create --name {env_name} python=3.7
conda install -n {env_name} matplotlib docopt ipython mpi4py
conda install -n {env_name} -c anaconda pyyaml
conda install -n {env_name} pytorch torchvision -c pytorch
conda install -n {env_name} tensorboardx -c conda-forge
```

## Models

All models should be located in `GANs_dir`. The class and the file that contains the class should have identic name. For example, class:
```
class CNN_model(GANs_abstract_object.GANs_model):
    ...
```
File:
```
CNN_model.py
```

