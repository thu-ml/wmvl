# A Wasserstein Minimum Velocity Approach to Learning Unnormalized Models

```
@InProceedings{wang20a,
  title = {A Wasserstein Minimum Velocity Approach to Learning Unnormalized Models},
  author = {Wang, Ziyu and Cheng, Shuyu and Yueru, Li and Zhu, Jun and Zhang, Bo},
  booktitle = {International Conference on Artificial Intelligence and Statistics},
  pages = {3728--3738},
  year = {2020},
  pdf = {https://arxiv.org/pdf/2002.07501.pdf},
}
```

## Requirements

Tensorflow 1.14, Python 3, [meta-inf/experiments](https://github.com/meta-inf/experiments)@`3938aa3c`. See `requirements.txt`.

You also need to execute `pip install -e .` in this directory.

To compute FID you need to download pretrained weights of [Inception-v3](https://ml.cs.tsinghua.edu.cn/~ziyu/static/wmvl/inception.tar) and [LeNet](https://ml.cs.tsinghua.edu.cn/~ziyu/static/wmvl/lenet.tar), and extract them into `~/{inception,lenet}`, respectively.

## Using the Code

```
cd src; python mnist.py
```

## Acknowledgement

This repository is based on the [official implementation of the hyperspherical VAE paper](https://github.com/nicola-decao/s-vae-tf). It also contains code adapted from [mbinkowski/MMD-GAN](https://github.com/mbinkowski/MMD-GAN), [bioinf-jku/TTUR](https://github.com/bioinf-jku/TTUR) and [jiamings/ais](https://github.com/jiamings/ais).

