#!/bin/bash

mkdir -p data/models/clip
mkdir -p data/models/imagenet

wget https://cmp.felk.cvut.cz/univ_emb/checkpoints/clip/vit_b16.npy -P data/models/clip/
wget https://cmp.felk.cvut.cz/univ_emb/checkpoints/imagenet/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz -P data/models/imagenet/
wget https://cmp.felk.cvut.cz/univ_emb/checkpoints/imagenet/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz -P data/models/imagenet/
