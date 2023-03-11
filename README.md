# Image-Denoising

Compared AE & KLM performance for denoising RGB images. 

Noisy images were created by adding Gassian noise $X \sim \mathcal{N}(\mu=0,\sigma=0.01)\$ to ground truth images.

Implemented with Python 3.9.

Create environment:

```
conda create -n <envname> python=3.9 anaconda
```

Activate virtual environment:

```
conda activate <envname>
```

Install required packages. For example, tensorflow-cpu:

```
conda install -c conda-forge tensorflow-cpu
```

Create a kernel in jupyter lab:

```
conda install ipykernel
ipython kernel install --user --name=<envname>
```
