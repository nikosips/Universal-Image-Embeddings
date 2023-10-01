import argparse
import json
import os
import random
import tensorflow as tf

from pathlib import Path




def load_split_info(info_file):
    with open(info_file, 'r') as infile:
        info_data = json.load(infile)

    return info_data


def pick_output_shard(num_shards):
    return random.randint(0, num_shards - 1)


def prepare_tfds_sharded(info_file, output_file, files_dir, num_shards):
    
    print(info_file, output_file, files_dir)
    info_data = load_split_info(info_file)
    
    random.shuffle(info_data) 
    

    writers = []
    for i in range(num_shards):
        writers.append(
            tf.io.TFRecordWriter("%s-%05i-of-%05i" % (output_file, i, num_shards)))

    for file_info in info_data:
        print('Processing: ' + file_info['path'])

        example = create_one_example(file_info['path'], file_info['class_id'], files_dir)
        writers[pick_output_shard(num_shards)].write(example.SerializeToString())

    # Close all files
    for w in writers:
        w.close()



def create_one_example(image_file, label, files_dir):
    
    if not isinstance(label, list):
        label = [label]
    
    feature = {
        "image_bytes": tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.read_file(os.path.join(files_dir, image_file)).numpy()])),
        "class_id": tf.train.Feature(int64_list=tf.train.Int64List(value=label)),
        "key": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_file.encode()])),
    }
    
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example


def prepare_tfds(info_file, output_file, files_dir):
    
    print(info_file, output_file, files_dir)
    info_data = load_split_info(info_file)
    
    with tf.io.TFRecordWriter(output_file) as writer:
        
        for file_info in info_data:
            print('Processing: ' + file_info['path'])
            
            example = create_one_example(file_info['path'], file_info['class_id'], files_dir)
            writer.write(example.SerializeToString())



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--info_file', dest='info_file',
                        help='Path to the json info file.')
    parser.add_argument('--output_file', dest='output_file',
                        help='Output tfds file.')
    parser.add_argument('--files_dir', dest='files_dir',
                        help='Directory to all the image files.')
    parser.add_argument('--num_shards', dest='num_shards', type=int,
                        help='Number of shards to be created.')
    parser.add_argument('--train', dest='train', action='store_true',
                       help='For train splits. Includes sharding if num_shards>1 and shuffling of the info_file')
    
    args = parser.parse_args()

    Path(args.output_file).parent.mkdir(parents=True,exist_ok=True)

    if args.train:
        prepare_tfds_sharded(args.info_file, args.output_file, args.files_dir, args.num_shards)
    else:
        prepare_tfds(args.info_file, args.output_file, args.files_dir)
    

if __name__ == '__main__':
    main()