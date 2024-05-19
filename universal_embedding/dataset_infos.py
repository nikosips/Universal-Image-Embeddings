DATASET_INFO = {
    'cars': {
        'domain': 0,
        'train_files': 'cars/train/cars.train.tfrecord',
        'test_files': 'cars/test/cars.test.tfrecord',
        'val_files': 'cars/val/cars.val.tfrecord',

        'num_train_classes': 78,
        'num_train_examples': 6346,
        'num_test_examples': 8131,
        'num_val_examples': 1708,
        'knn': {
            'train_knn': {
                'query': 'train',
                'index': 'train',
            },
            'val_knn': {
                'query': 'val',
                'index': 'val',
            },
            'test_knn': {
                'query': 'test',
                'index': 'test',
            },
        },
    },
    'sop': {
        'domain': 1,
        'train_files': 'sop/train/sop.train.tfrecord',
        'test_files': 'sop/test/sop.test.tfrecord',
        'val_files': 'sop/val/sop.val.tfrecord',

        'num_train_classes': 9054,
        'num_train_examples': 48942,
        'num_test_examples': 60502,
        'num_val_examples': 10609,
        'knn': {
            'train_knn': {
                'query': 'train',
                'index': 'train',
            },
            'val_knn': {
                'query': 'val',
                'index': 'val',
            },
            'test_knn': {
                'query': 'test',
                'index': 'test',
            },
        },
    },

    'inshop': {
        'domain': 2,
        'train_files': 'inshop/train/inshop.train.tfrecord',
        'test_files': 'inshop/test/inshop.test.tfrecord',
        'val_files': 'inshop/val/inshop.val.tfrecord',
        'index_files': 'inshop/index/inshop.index.tfrecord',  # size 12612

        'num_train_classes': 3198,
        'num_train_examples': 20897,
        'num_test_examples': 14218,
        'num_val_examples': 4982,
        'knn': {
            'train_knn': {
                'query': 'train',
                'index': 'train',
            },
            'val_knn': {
                'query': 'val',
                'index': 'val',
            },
            'test_knn': {
                'query': 'test',
                'index': 'index',
            },
        },
    },

    'inat': {
        'domain': 3,
        'train_files': 'inat/train/inat.train.tfrecord',
        'test_files': 'inat/test/inat.test.tfrecord',
        'val_files': 'inat/val/inat.val.tfrecord',

        'num_train_classes': 4552,
        'num_train_examples': 273929,
        'num_test_examples': 136093,
        'num_val_examples': 51917,
        'knn': {
            'train_knn': {
                'query': 'train',
                'index': 'train',
            },
            'val_knn': {
                'query': 'val',
                'index': 'val',
            },
            'test_knn': {
                'query': 'test',
                'index': 'test',
            },
        },
    },

    'met': {
        'domain': 4,
        'train_files': 'met/train/met.train.tfrecord',
        'small_train_files': 'met/small_train/met.small_train.tfrecord',  # size 38307
        'test_files': 'met/test/met.test.tfrecord',
        'val_files': 'met/val/met.val.tfrecord',

        'index_files': 'met/index/met.index.tfrecord', #same as train set
        'small_index_files': 'met/small_index/met.small_index.tfrecord', #same as small train set

        'num_train_classes': 224408,
        'num_train_examples': 397121,
        'num_test_examples': 1003,
        'num_val_examples': 129,
        'knn': {
            'train_knn': {
                'query': 'train',
                'index': 'train',
            },
            'val_knn': {
                'query': 'val',
                'index': 'small_index',
            },
            'test_knn': {
                'query': 'test',
                'index': 'index',
            },
        },
    },

    'gldv2': {
        'domain': 5,
        'train_files': 'gldv2/train/gldv2.train.tfrecord',
        'test_files': 'gldv2/test/gldv2.test.tfrecord',
        'val_files': 'gldv2/val/gldv2.val.tfrecord',
        'index_files': 'gldv2/index/gldv2.index.tfrecord',  # size 761757

        'num_train_classes': 73182,
        'num_train_examples': 1422914,
        'num_test_examples': 1129,
        'num_val_examples': 157556,
        'knn': {
            'train_knn': {
                'query': 'train',
                'index': 'train',
            },
            'val_knn': {
                'query': 'val',
                'index': 'val',
            },
            'test_knn': {
                'query': 'test',
                'index': 'index',
            },
        },
    },

    'food2k': {
        'domain': 6,
        'train_files': 'food2k/train/food2k.train.tfrecord',
        'test_files': 'food2k/test/food2k.test.tfrecord',
        'val_files': 'food2k/val/food2k.val.tfrecord',


        'num_train_classes': 900,
        'num_train_examples': 472349,
        'num_test_examples': 9979,
        'num_val_examples': 49323,
        'knn': {
            'train_knn': {
                'query': 'train',
                'index': 'train',
            },
            'val_knn': {
                'query': 'val',
                'index': 'val',
            },
            'test_knn': {
                'query': 'test',
                'index': 'test',
            },
        },
    },

    'rp2k': {
        'domain': 7,
        'train_files': 'rp2k/train/rp2k.train.tfrecord',
        'test_files': 'rp2k/test/rp2k.test.tfrecord',
        'val_files': 'rp2k/val/rp2k.val.tfrecord',


        'num_train_classes': 1074,
        'num_train_examples': 188724,
        'num_test_examples': 10931,
        'num_val_examples': 17185,
        'knn': {
            'train_knn': {
                'query': 'train',
                'index': 'train',
            },
            'val_knn': {
                'query': 'val',
                'index': 'val',
            },
            'test_knn': {
                'query': 'test',
                'index': 'test',
            },
        },
    },
}