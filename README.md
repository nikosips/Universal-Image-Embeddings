# Official Repository for the UnED dataset and benchmark

This repository supports training and evaluating Universal Embeddings.
The experiments in the respective publication can be replicated using the code provided here.
The implementation uses [Scenic](https://github.com/google-research/scenic) framework, building on top of [JAX](https://github.com/google/jax) and [Flax](https://github.com/google/flax).

The repository will be constantly updated for some time, in order to provide an easier interface for training and evaluation.
New features will be added soon to make the use of the UnED dataset easier, as well as to improve ease of evaluating models on it.

Project webpage (includes UnED splits):
https://cmp.felk.cvut.cz/univ_emb/

Requirements:

1) Make sure to use Python 3.9
2) Create Virtual Environment: ```python3.9 -m venv scenic_venv```
3) Activate the Virtual Environment: ```source scenic_venv/bin/activate```
4) Clone scenic: ```git clone https://github.com/google-research/scenic.git```
5) Install scenic dependencies: ```cd scenic && pip install .```
Keep in mind that you might need to modify the installation of Jax used depending on your accelerator.
6) clone our repo: ```git clone https://github.com/nikosips/Universal-Image-Embeddings.git```
7) ```cd Universal-Image-Embeddings```


## Dataset preparation

The UnED dataset contains 8 existing datasets.
We provide a way to prepare all the data for training and evaluation.


1. Create "data" directory.

    * Download (links for the images are in the project website) and extract images into data/images directory.


    * Download and extract info files from dataset website like this:

```
├── info_files
│   ├── cars
│   │   ├── test.json
│   │   ├── train.json
│   │   └── val.json
│   ├── food2k
│   │   ├── test.json
│   │   ├── train.json
│   │   └── val.json
│   ├── gldv2
│   │   ├── index.json
│   │   ├── test.json
│   │   ├── train.json
│   │   └── val.json
│   ├── inat
│   │   ├── test.json
│   │   ├── train.json
│   │   └── val.json
│   ├── inshop
│   │   ├── index.json
│   │   ├── test.json
│   │   ├── train.json
│   │   └── val.json
│   ├── met
│   │   ├── index.json
│   │   ├── small_index.json
│   │   ├── small_train.json
│   │   ├── test.json
│   │   ├── train.json
│   │   └── val.json
│   ├── rp2k
│   │   ├── test.json
│   │   ├── train.json
│   │   └── val.json
│   └── sop
│       ├── test.json
│       ├── train.json
│       └── val.json
```


#### The provided Scenic code trains and evaluates with tfrecord dataloaders (tfds). These tfds files are hardcoded in "datasets.py" script, in DATASET_INFO dictionary, for each domain of the UnED dataset.

To create them from the image files and the info files, use the provided "convert_to_tfrecord_from_file.py" script.
This needs to be done for every split in the UnED dataset (every .json file).

Example usage: 

for training splits of each domain:

```
dataset=gldv2; split=train ; python convert_to_tfrecord_from_file.py --info_file ~/data/info_files/$dataset/$split.json --output_file ~/data/tfds/$dataset/$split/$dataset.$split.tfrecord --files_dir ~/data/images/ --num_shards 150 --train
```

for evaluation splits of each domain (splits of val and test set):

```
dataset=met; split=test ; python convert_to_tfrecord_from_file.py --info_file ~/data/info_files/$dataset/$split.json --output_file ~/data/tfds/$dataset/$split/$dataset.$split.tfrecord --files_dir ~/data/images/ --num_shards 1
```

- - - -

### For training and evaluation, download the pretrained models used in this work.
All training and evaluation hyperparameters are set in the config files.
Sample config files are inside the universal_embedding/configs directory.

Checkpoints used in our work are hosted in the website.

In all the config files, you have to provide where info files are, as they are used in evaluation, as well as the directory of the checkpoints and the tfds files.

Click to expand the use cases of our code below:

<details>

  <summary><b>Model training and validation (and optional final testing)</b></summary><br/>

  Configure the "config_train_clip_vit.py" or "config_train_vit.py" to the type of training you want to perform.
  Checkpoints, descriptors and event files are saved in ```YOUR_WORKDIR```.

  ```
  python -m universal_embedding.main --config=universal_embedding/configs/config_train_clip_vit.py --workdir=YOUR_WORKDIR
  ```

</details>

<details>

  <summary><b>Descriptor extraction and evaluation</b></summary><br/>

  Configure the "config_knn_clip_vit.py" or "config_knn_vit.py" to the type of evaluation you want to perform.
  Pro
  Event files are saved in ```YOUR_WORKDIR```.

  ```
  python -m universal_embedding.knn_main --config=universal_embedding/configs/config_knn_clip_vit.py --workdir=YOUR_WORKDIR
  ```

</details>

<details>

  <summary><b>Descriptor evaluation</b></summary><br/>

  Configure the "config_knn_descr_eval.py" to the type of evaluation you want to perform.
  Event files are saved in ```YOUR_WORKDIR```.

  ```
  python -m universal_embedding.descr_eval --config=universal_embedding/configs/config_knn_descr_eval.py --workdir=YOUR_WORKDIR
  ```

</details>


- - - -

## Citation

If you use our work in yours, please cite us using the following:

```
@InProceedings{Ypsilantis_2023_ICCV,
    author    = {Ypsilantis, Nikolaos-Antonios and Chen, Kaifeng and Cao, Bingyi and Lipovsk\'y, M\'ario and Dogan-Sch\"onberger, Pelin and Makosa, Grzegorz and Bluntschli, Boris and Seyedhosseini, Mojtaba and Chum, Ond\v{r}ej and Araujo, Andr\'e},
    title     = {Towards Universal Image Embeddings: A Large-Scale Dataset and Challenge for Generic Image Representations},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {11290-11301}
}
```

- - - -