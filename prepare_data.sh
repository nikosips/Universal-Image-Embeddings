#!/bin/bash

dataset=cars; split=train ; python convert_to_tfrecord_from_file.py --info_file data/info_files/$dataset/$split.json --output_file data/tfds/$dataset/$split/$dataset.$split.tfrecord --files_dir data/images --num_shards 1 --train
dataset=cars; split=val ; python convert_to_tfrecord_from_file.py --info_file data/info_files/$dataset/$split.json --output_file data/tfds/$dataset/$split/$dataset.$split.tfrecord --files_dir data/images --num_shards 1
dataset=cars; split=test ; python convert_to_tfrecord_from_file.py --info_file data/info_files/$dataset/$split.json --output_file data/tfds/$dataset/$split/$dataset.$split.tfrecord --files_dir data/images --num_shards 1

dataset=food2k; split=train ; python convert_to_tfrecord_from_file.py --info_file data/info_files/$dataset/$split.json --output_file data/tfds/$dataset/$split/$dataset.$split.tfrecord --files_dir data/images --num_shards 50 --train
dataset=food2k; split=val ; python convert_to_tfrecord_from_file.py --info_file data/info_files/$dataset/$split.json --output_file data/tfds/$dataset/$split/$dataset.$split.tfrecord --files_dir data/images --num_shards 1
dataset=food2k; split=test ; python convert_to_tfrecord_from_file.py --info_file data/info_files/$dataset/$split.json --output_file data/tfds/$dataset/$split/$dataset.$split.tfrecord --files_dir data/images --num_shards 1

dataset=gldv2; split=train ; python convert_to_tfrecord_from_file.py --info_file data/info_files/$dataset/$split.json --output_file data/tfds/$dataset/$split/$dataset.$split.tfrecord --files_dir data/images --num_shards 150 --train
dataset=gldv2; split=val ; python convert_to_tfrecord_from_file.py --info_file data/info_files/$dataset/$split.json --output_file data/tfds/$dataset/$split/$dataset.$split.tfrecord --files_dir data/images --num_shards 1
dataset=gldv2; split=test ; python convert_to_tfrecord_from_file.py --info_file data/info_files/$dataset/$split.json --output_file data/tfds/$dataset/$split/$dataset.$split.tfrecord --files_dir data/images --num_shards 1
dataset=gldv2; split=index ; python convert_to_tfrecord_from_file.py --info_file data/info_files/$dataset/$split.json --output_file data/tfds/$dataset/$split/$dataset.$split.tfrecord --files_dir data/images --num_shards 1

dataset=inat; split=train ; python convert_to_tfrecord_from_file.py --info_file data/info_files/$dataset/$split.json --output_file data/tfds/$dataset/$split/$dataset.$split.tfrecord --files_dir data/images --num_shards 30 --train
dataset=inat; split=val ; python convert_to_tfrecord_from_file.py --info_file data/info_files/$dataset/$split.json --output_file data/tfds/$dataset/$split/$dataset.$split.tfrecord --files_dir data/images --num_shards 1
dataset=inat; split=test ; python convert_to_tfrecord_from_file.py --info_file data/info_files/$dataset/$split.json --output_file data/tfds/$dataset/$split/$dataset.$split.tfrecord --files_dir data/images --num_shards 1

dataset=inshop; split=train ; python convert_to_tfrecord_from_file.py --info_file data/info_files/$dataset/$split.json --output_file data/tfds/$dataset/$split/$dataset.$split.tfrecord --files_dir data/images --num_shards 2 --train
dataset=inshop; split=val ; python convert_to_tfrecord_from_file.py --info_file data/info_files/$dataset/$split.json --output_file data/tfds/$dataset/$split/$dataset.$split.tfrecord --files_dir data/images --num_shards 1
dataset=inshop; split=test ; python convert_to_tfrecord_from_file.py --info_file data/info_files/$dataset/$split.json --output_file data/tfds/$dataset/$split/$dataset.$split.tfrecord --files_dir data/images --num_shards 1
dataset=inshop; split=index ; python convert_to_tfrecord_from_file.py --info_file data/info_files/$dataset/$split.json --output_file data/tfds/$dataset/$split/$dataset.$split.tfrecord --files_dir data/images --num_shards 1

dataset=met; split=train ; python convert_to_tfrecord_from_file.py --info_file data/info_files/$dataset/$split.json --output_file data/tfds/$dataset/$split/$dataset.$split.tfrecord --files_dir data/images --num_shards 40 --train
dataset=met; split=val ; python convert_to_tfrecord_from_file.py --info_file data/info_files/$dataset/$split.json --output_file data/tfds/$dataset/$split/$dataset.$split.tfrecord --files_dir data/images --num_shards 1
dataset=met; split=small_index ; python convert_to_tfrecord_from_file.py --info_file data/info_files/$dataset/$split.json --output_file data/tfds/$dataset/$split/$dataset.$split.tfrecord --files_dir data/images --num_shards 1
dataset=met; split=test ; python convert_to_tfrecord_from_file.py --info_file data/info_files/$dataset/$split.json --output_file data/tfds/$dataset/$split/$dataset.$split.tfrecord --files_dir data/images --num_shards 1
dataset=met; split=index ; python convert_to_tfrecord_from_file.py --info_file data/info_files/$dataset/$split.json --output_file data/tfds/$dataset/$split/$dataset.$split.tfrecord --files_dir data/images --num_shards 1

dataset=rp2k; split=train ; python convert_to_tfrecord_from_file.py --info_file data/info_files/$dataset/$split.json --output_file data/tfds/$dataset/$split/$dataset.$split.tfrecord --files_dir data/images --num_shards 20 --train
dataset=rp2k; split=val ; python convert_to_tfrecord_from_file.py --info_file data/info_files/$dataset/$split.json --output_file data/tfds/$dataset/$split/$dataset.$split.tfrecord --files_dir data/images --num_shards 1
dataset=rp2k; split=test ; python convert_to_tfrecord_from_file.py --info_file data/info_files/$dataset/$split.json --output_file data/tfds/$dataset/$split/$dataset.$split.tfrecord --files_dir data/images --num_shards 1

dataset=sop; split=train ; python convert_to_tfrecord_from_file.py --info_file data/info_files/$dataset/$split.json --output_file data/tfds/$dataset/$split/$dataset.$split.tfrecord --files_dir data/images --num_shards 5 --train
dataset=sop; split=val ; python convert_to_tfrecord_from_file.py --info_file data/info_files/$dataset/$split.json --output_file data/tfds/$dataset/$split/$dataset.$split.tfrecord --files_dir data/images --num_shards 1
dataset=sop; split=test ; python convert_to_tfrecord_from_file.py --info_file data/info_files/$dataset/$split.json --output_file data/tfds/$dataset/$split/$dataset.$split.tfrecord --files_dir data/images --num_shards 1