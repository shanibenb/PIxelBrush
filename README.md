# PIxelBrush

## Intoduction
This is a PyTorch implementaion for a tool called PixelBrush that generates an artistic image using a given description.
This implementation based on [Jaile's work](http://cs231n.stanford.edu/reports/2017/pdfs/322.pdf) and a PyTorch implemention of
[Generative Adversarial Text-to-Image Synthesis paper](https://github.com/aelnouby/Text-to-Image-Synthesis).

## Requirements
- pytorch 
- visdom
- h5py
- PIL
- numpy
- skip-thoughts

## Dataset
We used the images from [The Oxford Paintings Dataset](http://www.robots.ox.ac.uk/~vgg/data/paintings/). 
We creating descriptions using [Neuraltalk2](https://github.com/raoyongming/neuraltalk2.pytorch).

The images names and descriptions we use are in 
[data/oxford/vis_oxford.json](https://github.com/shanibenb/PIxelBrush/blob/master/data/oxford/vis_oxford.json) file.

## Training
1. Create text embedding using [skip-thoughts](https://github.com/Cadene/skip-thoughts.torch/tree/master/pytorch),
and the file [create_embedding_oxford_uniskip_biskip.py](https://github.com/shanibenb/PIxelBrush/blob/master/create_embedding_oxford_uniskip_biskip.py). 
(You can skip this step by using the files in data/oxford).
2. Download the oxford dataset.
3. Create hd5 dataset using [convert_oxford_to_hd5_script.py](https://github.com/shanibenb/PIxelBrush/blob/master/convert_oxford_to_hd5_script.py)
(You can skip this step by downloading hd5 files from [drive](https://drive.google.com/drive/folders/1gyFgwRgJaQbRLWXaD3c8-xXRWbWAT3Ct?usp=sharing)).
4. run runtime.py

## Testing
1. Download hd5 files from [drive](https://drive.google.com/drive/folders/1gyFgwRgJaQbRLWXaD3c8-xXRWbWAT3Ct?usp=sharing).
2. run runtime.py with the following parameters:
- type: choose an architecture (simple | normal | deep)
- inference = True
- split = 10
- pre_trained_gen = "pre-trained-models/{}_200.pth"
