"""Script for Knn evalulation."""
import functools

from clu import metric_writers

from absl import logging
from flax import jax_utils
import jax
import jax.numpy as jnp
import ml_collections

from scenic.train_lib import train_utils

import os
import sys

from universal_embedding import app

from universal_embedding import train_eval_steps
from universal_embedding import datasets
from universal_embedding import knn_utils
from universal_embedding import models
from universal_embedding import model_init
from universal_embedding import utils


def extract_dir_descriptors(
	rng: jnp.ndarray,
	config: ml_collections.ConfigDict,
	workdir: str,
	writer: metric_writers.MetricWriter,
) -> None:

	lead_host = jax.process_index() == 0


	### To remove code
	my_list_of_paths_flag=True

	if my_list_of_paths_flag:

		####################
		with open(os.path.join(config.base_dir, 'test_gallery.txt')) as fid:
		#with open(os.path.join(config.base_dir, 'test_query.txt')) as fid:    
			db_lines = fid.read().splitlines()

		db_lines = [os.path.join(config.base_dir, x.split(',')[0]) for x in db_lines] 

		list_of_paths=db_lines

	else:

		list_of_paths=None

	###


	dataset_dict = datasets.get_extract_dataset(
		config,
		base_dir=config.base_dir,
		list_of_paths=list_of_paths, #make it work with list of paths instead if given properly #one has to take care of it himself
		eval_batch_size=config.get('eval_batch_size', config.batch_size),
	)

	model_cls = models.MODELS[config.model_class]
	model = model_cls(config, dataset_dict.meta_data)

	rng, init_rng = jax.random.split(rng)

	(params, model_state, num_trainable_params, gflops) = model_init.initialize_universal_model_for_extraction(
		dataset_dict,
		config,
		model,
		init_rng,
	)

	train_state = train_utils.TrainState(
		params=params,
		model_state=model_state
	)
	
	if config.pretrained_ckpt:

		logging.info('Initializing from ckpt %s.', config.pretrained_ckpt)

		if config.model_class.startswith("clip_vit_with_embedding"):

			train_state = model.load_model_vars(
				train_state, config.pretrained_ckpt,
			)

		elif config.model_class.startswith("vit_with_embedding"):

			train_state = model.load_augreg_params(
				train_state, config.pretrained_ckpt, config.model
			)
	
	train_state = train_state.replace(metadata={})
	train_state = jax_utils.replicate(train_state)

	del params

	#project feats or not
	representation_fn_knn = functools.partial(
		train_eval_steps.representation_fn_eval,
		flax_model = model.flax_model,
		project_feats = config.project_feats_knn,
		config=config,
	)

	knn_eval_batch_size = config.get('knn_eval_batch_size') or config.batch_size

	knn_evaluator = knn_utils.KNNEvaluator(
		config,
		representation_fn_knn,
		knn_eval_batch_size,
		config.get("extract_only_descrs",False),
	)

	train_dir = config.get('train_dir')

	if config.test_pretrained_features:

		descriptors = knn_evaluator._get_repr(      
			train_state,
			dataset_dict.extract_iter,
		)

	if config.only_best_knn:

		step = -1

		print(f"step: {step}")

		if not config.preextracted:
			ckpt_file = os.path.join(train_dir,str(step))
			ckpt_info = ckpt_file.split('/')
			ckpt_dir = '/'.join(ckpt_info[:-1])
			ckpt_num = ckpt_info[-1].split('_')[-1]

			try:

				train_state, _ = train_utils.restore_checkpoint(
					ckpt_dir,
					train_state,
					assert_exist=True,
					step=int(ckpt_num),
				)

			except:

				sys.exit("no checkpoint found")

			train_state = train_state.replace(metadata={})
			train_state = jax_utils.replicate(train_state)

		else:

			train_state = None

		descriptors = knn_evaluator._get_repr(      
			train_state, 
			dataset_dict.extract_iter,
		)

	else:

		for epoch in range(config.knn_start_epoch,config.knn_end_epoch+1):

			step = epoch * config.steps_per_epoch

			print(f"step: {step}")

			if not config.preextracted:
				ckpt_file = os.path.join(train_dir,str(step))
				ckpt_info = ckpt_file.split('/')
				ckpt_dir = '/'.join(ckpt_info[:-1])
				ckpt_num = ckpt_info[-1].split('_')[-1]

				try:

					train_state, _ = train_utils.restore_checkpoint(
						ckpt_dir,
						train_state,
						assert_exist=True,
						step=int(ckpt_num),
					)

				except:

					sys.exit("no checkpoint found")

				train_state = train_state.replace(metadata={})
				train_state = jax_utils.replicate(train_state)

			else:

				train_state = None

			descriptors = knn_evaluator._get_repr(      
				train_state, 
				dataset_dict.extract_iter,
			)


	descriptors_dict = {
		'descriptors': descriptors[0],
		'paths': dataset_dict.meta_data['paths']
	}

	#save descriptors here
	utils.save_descriptors(workdir+"/descriptors.json",descriptors_dict)

	train_utils.barrier_across_hosts() #is this even needed?



if __name__ == '__main__':
	app.run(main=extract_dir_descriptors,knn=True)