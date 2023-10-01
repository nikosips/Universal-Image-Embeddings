"""Utils for K-nearest neighbor evaluation."""

import collections
import copy
from absl import logging
from clu import metric_writers
from flax import jax_utils
import jax
import numpy as np
import jax.numpy as jnp

import functools
import json
from tensorflow.io import gfile
import os

from universal_embedding import datasets
from universal_embedding import utils




class KNNEvaluator:
  """Class for knn evaluation."""

  def __init__(
    self,
    config,
    representation_fn,
    knn_query_batch_size,
    extract_only_descriptors = False,
  ):

    self.config = config
    self.extract_only_descriptors = extract_only_descriptors

    if representation_fn is not None:
      self.repr_fn = jax.pmap(
        representation_fn, 
        donate_argnums=(1,), 
        axis_name='batch',
      )

    else:
      self.repr_fn = None

    self.knn_query_batch_size = knn_query_batch_size
    self.compute_knn_metrics_fun = self.compute_knn_metrics



  @staticmethod
  def _run_knn(
    k,
    index_descrs, 
    query_descrs
  ):

    all_similarities = jnp.matmul(query_descrs, jnp.transpose(index_descrs))
    similarities_k_sorted, indices_k_sorted = jax.lax.top_k(all_similarities, k)

    return similarities_k_sorted,indices_k_sorted



  def _get_repr(
    self, 
    train_state, 
    data,
  ):

    """Compute representation for a dataset split."""
  
    embedding_list = []

    print("extracting representations")

    for batch in data:

      embeddings, mask = self.repr_fn(train_state, batch)
      
      # We need to unreplicate the output of `lax.all_gather`.
      # Shapes at this point are:
      #   embedding: `[hosts, devices, global_batch, features]`.
      mask = np.array(jax_utils.unreplicate(mask)).astype(bool)
      embedding_list.append(np.array(jax_utils.unreplicate(embeddings))[mask])
      
    embedding = np.concatenate(embedding_list, axis=0)

    print("extracted representations")
  
    return [embedding]



  def compute_knn_metrics(
    self,
    lookup_key,
    query_results, 
    index_results, 
    query_paths,
    index_paths,
    throw_first, 
    top_k, 
    config=None,
    query_labels=None,
    index_labels=None,
    query_domains=None,
    index_domains=None,    
  ):
    """Compute knn metrics on the query and index."""

    results_visuals = []

    actual_top_k = top_k
    knn_top_k = 1
    
    if throw_first:
      knn_top_k += 1
      top_k += 1

    query_emb = query_results[0]
    index_emb = index_results[0]

    num_query = len(query_emb)

    query_emb = np.array(query_emb)
    index_emb, index_labels = np.array(index_emb), np.array(index_labels)

    logging.info(f'num query embedding: {num_query}')
    logging.info(f'num index embedding: {len(index_emb)}')
    logging.info(f'embedding dimension: {query_emb.shape[-1]}')

    assert len(np.unique(query_domains)) == 1

    index_label_counter = collections.Counter(index_labels[np.where(index_domains == np.unique(query_domains)[0])[0]])

    num_batch = num_query // self.knn_query_batch_size

    if num_query % self.knn_query_batch_size != 0:
      num_batch += 1

    logging.info('num_eval_batch: %d', num_batch)
    num_knn_correct = 0
    mp = 0.0
    ap = 0.0

    pmapped_clf_predict = jax.pmap(
      functools.partial(
        self._run_knn,
        k = top_k,
        index_descrs = index_emb,
      )
    )

    relevances_map_all = []

    for i in range(num_batch):
      
      batch_queries = query_emb[i* self.knn_query_batch_size : min(
                            (i + 1) * self.knn_query_batch_size, num_query)]

      array_batches,masks = self.split_and_pad(batch_queries)
      masks = masks.astype(bool)

      similarities_k_sorted, indices_k_sorted = pmapped_clf_predict(query_descrs = array_batches)      

      similarities_k_sorted = np.array(similarities_k_sorted[masks])
      indices_k_sorted = np.array(indices_k_sorted[masks])
      
      predicted_positions = indices_k_sorted

      for k in range(
          i * self.knn_query_batch_size,
          min((i + 1) * self.knn_query_batch_size, num_query),
      ): #for every query in the batch of queries
        
        m = k - (i * self.knn_query_batch_size)

        nearest = [
            (index_labels[j], index_domains[j], similarities_k_sorted[m,l]) #was j instead of l
            for l,j in enumerate(predicted_positions[m])
        ]
        
        #R@1 calc
        pred_label, pred_domain = (
            nearest[knn_top_k - 1][0],
            nearest[knn_top_k - 1][1],
        )

        query_label, query_domain = query_labels[k], query_domains[k]

        if np.any(pred_label == query_label) and pred_domain == query_domain:
          num_knn_correct += 1

        #mMP@5 calc
        num_correct = 0

        if isinstance(query_label,int):          
          num_index_label = index_label_counter[query_label]

        else:
          num_index_label = np.sum(
              [index_label_counter[label] for label in query_label]
          )
        
        # num_index_label == how many from the same class are in the index
        # if the query set is the index set this value is +1 of the correct
        # that must be used as n_q in the definition
        if throw_first:
          num_true_index_label = num_index_label - 1
        else:
          num_true_index_label = num_index_label

        relevances = [] #just for visual
        relevances_map = []

        for j in range(min(top_k,num_index_label)):
          
          #for every neighbor
          if np.any(query_label == nearest[j][0]) and query_domain == nearest[j][1]:
            num_correct += 1
            relevances.append(1)
          else:
            relevances.append(0)

        #for loop for map@k
        for j in range(top_k):
          
          #for every neighbor
          if np.any(query_label == nearest[j][0]) and query_domain == nearest[j][1]:
            relevances_map.append(1)
          else:
            relevances_map.append(0)

        # Remove the offset.
        if throw_first:
          num_correct -= 1
          relevances = relevances[1:]

          relevances_map = relevances_map[1:]

        ##mmp@k calc
        mp += (num_correct * 1.0) / min(num_true_index_label, actual_top_k)

        ##map@k calc
        assert len(relevances_map) == actual_top_k

        relevances_map_all.append(relevances_map)

        prec = np.cumsum(relevances_map) / (1+np.array(np.arange(len(relevances_map))))
        ap += ((prec * relevances_map).sum()) / min(num_true_index_label, actual_top_k)

        #save list of neighbors of each query
        if config.save_neighbors:

          query_path = query_paths[k]
          neighbor_indices = predicted_positions[m]

          if throw_first:
            nearest = nearest[1:]
            neighbor_indices = neighbor_indices[1:]

          #stop seeing neighbors after num_true_index_label or 5
          neighbor_paths = [index_paths[x] for x in neighbor_indices][:num_true_index_label]
          neighbor_classes = [x[0] for x in nearest][:num_true_index_label]
          neighbor_domains = [x[1] for x in nearest][:num_true_index_label]
          neighbor_simils = [x[2] for x in nearest][:num_true_index_label]

          assert len(neighbor_domains) == len(neighbor_classes) == len(neighbor_paths) == len(neighbor_simils)
          
          all_paths = [query_path]
          all_paths.extend(neighbor_paths)

          query_domain_name = datasets.invert_DOMAIN_LABELS[query_domain]
          neighbor_domains = [datasets.invert_DOMAIN_LABELS[x] for x in neighbor_domains]

          relevances.insert(0,-1)

          query_dict = {}
          query_dict["paths"] = all_paths
          query_dict["query_domain_name"] = query_domain_name
          query_dict["neighbor_domains"] = neighbor_domains
          query_dict["neighbor_classes"] = neighbor_classes
          query_dict["relevances"] = relevances

          results_visuals.append(query_dict)

    mean_acc = np.round(np.array(num_knn_correct * 1.0 / num_query),3)
    mean_mmp_at_k = np.round(np.array(mp / num_query),3)
    mean_map_at_k = np.round(np.array(ap / num_query),3)

    return mean_acc, mean_mmp_at_k, mean_map_at_k, results_visuals



  def run_separate_knn(
    self,
    train_state,
    base_dir,
    dataset_names,
    batch_size,
    disabled_knns='',
    all_descriptors_dict = None,
    config = None,
  ):

    results_visuals_all = {}

    """Runs seperate knn evals defined in the dataset."""
    dataset = datasets.get_knn_eval_datasets(
      self.config,
      base_dir, 
      dataset_names.split(','), 
      batch_size,
    )

    knn_info = dataset.knn_info
    
    knn_results, mp_results, map_results = {}, {}, {}
    knn_name_avg,mp_name_avg,map_name_avg = {}, {}, {}
    

    if all_descriptors_dict is None:
      all_descriptors_dict = {}

    for dataset_name, info in knn_info['knn_setup'].items(): #for every dataset  
     
      if dataset_name in all_descriptors_dict:
        inference_lookup = all_descriptors_dict[dataset_name]
      else:
        inference_lookup = {}


      for knn_name, val in info.items(): 

        if knn_name in disabled_knns:
          logging.info(
              'Skipping disabled knn %s in separate knn eval.', knn_name
          )
          continue

        for split in [val['query'], val['index']]:
          #so that we do not extract twice for the same split
          lookup_key = datasets.dataset_lookup_key(dataset_name, split)
          
          if split not in inference_lookup and train_state is not None:
            logging.info('Getting inference for %s.', lookup_key)
            inference_lookup[split] = self._get_repr(
                train_state, 
                knn_info[lookup_key],
            )
          else:
            print(f"descriptors already extracted for {lookup_key}")

      if dataset_name in all_descriptors_dict:
        all_descriptors_dict[dataset_name].update(inference_lookup)
      else:
        all_descriptors_dict[dataset_name] = inference_lookup

      if self.extract_only_descriptors:
        continue

      for knn_name, val in info.items():

        if knn_name in disabled_knns:
          logging.info(
              'Skipping disabled knn %s in separate knn eval', knn_name
          )
          continue
        
        logging.info(
            'Running knn on dataset %s with split %s in separate knn eval.',
            dataset_name,
            knn_name,
        )
        
        query_split = val['query']
        index_split = val['index']

        query_labels = knn_info['json_data'][datasets.dataset_lookup_key(dataset_name, query_split)]["labels"]
        index_labels = knn_info['json_data'][datasets.dataset_lookup_key(dataset_name, index_split)]["labels"]

        query_domains = knn_info['json_data'][datasets.dataset_lookup_key(dataset_name, query_split)]["domains"]
        index_domains = knn_info['json_data'][datasets.dataset_lookup_key(dataset_name, index_split)]["domains"]

        query_paths = knn_info['json_data'][datasets.dataset_lookup_key(dataset_name, query_split)]["paths"]
        index_paths = knn_info['json_data'][datasets.dataset_lookup_key(dataset_name, index_split)]["paths"]

        logging.info(
            'query_split: %s, index_split: %s.', query_split, index_split
        )
        
        throw_1st = True if query_split == index_split else False
        
        logging.info('throw_1st: %s.', throw_1st)

        (
          knn_results[dataset_name + ':separate:' + knn_name + ':top_1'],
          mp_results[dataset_name + ':separate:' + knn_name + f':mp_{dataset.meta_data["top_k"]}'],
          map_results[dataset_name + ':separate:' + knn_name + f':map_{dataset.meta_data["top_k"]}'],
          results_visuals
        ) = self.compute_knn_metrics_fun(
          datasets.dataset_lookup_key(dataset_name, query_split),
          inference_lookup[query_split],
          inference_lookup[index_split],
          query_paths,
          index_paths,
          throw_1st,
          dataset.meta_data['top_k'],
          config,
          query_labels,
          index_labels,
          query_domains,
          index_domains,
        )

        results_visuals_all[f"{dataset_name}:{query_split}"] = results_visuals

        if knn_name not in knn_name_avg:
          knn_name_avg[knn_name] = {}
        knn_name_avg[knn_name][dataset_name] = knn_results[dataset_name + ':separate:' + knn_name + ':top_1']

        knn_results['average:separate:' + knn_name + ':top_1'] = \
            np.round(np.mean([knn_name_avg[knn_name][dataset_name] for dataset_name in knn_name_avg[knn_name].keys()]),3)

        if knn_name not in mp_name_avg:
          mp_name_avg[knn_name] = {}
        mp_name_avg[knn_name][dataset_name] = mp_results[dataset_name + ':separate:' + knn_name + f':mp_{dataset.meta_data["top_k"]}']

        mp_results['average:separate:' + knn_name + f':mp_{dataset.meta_data["top_k"]}'] = \
            np.round(np.mean([mp_name_avg[knn_name][dataset_name] for dataset_name in mp_name_avg[knn_name].keys()]),3)

        if knn_name not in map_name_avg:
          map_name_avg[knn_name] = {}
        map_name_avg[knn_name][dataset_name] = map_results[dataset_name + ':separate:' + knn_name + f':map_{dataset.meta_data["top_k"]}']

        map_results['average:separate:' + knn_name + f':map_{dataset.meta_data["top_k"]}'] = \
            np.round(np.mean([map_name_avg[knn_name][dataset_name] for dataset_name in map_name_avg[knn_name].keys()]),3)

    return [knn_results, mp_results, map_results], all_descriptors_dict, results_visuals_all



  def run_merged_knn(
    self,
    train_state,
    base_dir,
    query_dataset_names,
    index_dataset_names,
    batch_size,
    disabled_knns='',
    all_descriptors_dict = None,
    config = None,
  ):

    results_visuals_all = {}

    """Runs  knn evals using a common database."""
    query_dataset_names = set(query_dataset_names.split(','))
    index_dataset_names = set(index_dataset_names.split(','))
    
    assert query_dataset_names.issubset(
        index_dataset_names
    ), 'Please make sure query set names are a subset of index set names.'

    dataset_names = query_dataset_names.union(index_dataset_names)

    dataset = datasets.get_knn_eval_datasets(
      self.config,
      base_dir,
      list(dataset_names),
      batch_size,
    )
    
    knn_info = dataset.knn_info
    knn_results,mp_results,map_results = {}, {}, {}

    knn_name_avg,mp_name_avg,map_name_avg = {}, {}, {}

    lookup_keys = set()

    for dataset_name, info in knn_info['knn_setup'].items(): #for each dataset
    
      for knn_name, val in info.items(): #for each split
        
        if knn_name in disabled_knns:
          logging.info('Skipping disabled knn %s in merged knn eval', knn_name)
          continue

        if dataset_name in index_dataset_names:
          lookup_key = datasets.dataset_lookup_key(dataset_name, val['index'])
          lookup_keys.add(lookup_key)
        if dataset_name in query_dataset_names:
          lookup_key = datasets.dataset_lookup_key(dataset_name, val['query'])
          lookup_keys.add(lookup_key)

        if config.do_pcaw:
          lookup_key = datasets.dataset_lookup_key(dataset_name, val['pcaw'])

    inference_lookup = {}
  
    if all_descriptors_dict is not None:
      for dataset_name in all_descriptors_dict.keys():
        for split in all_descriptors_dict[dataset_name].keys():
          inference_lookup[f"{dataset_name}:{split}"] = all_descriptors_dict[dataset_name][split]



    for lookup_key in lookup_keys:
      
      if lookup_key not in inference_lookup and train_state is not None:

        logging.info('Getting inference for %s in merged knn eval.', lookup_key)
        inference_lookup[lookup_key] = self._get_repr(
            train_state, 
            knn_info[lookup_key],
        )

      else:
        print(f"representations already extracted for {lookup_key}")

        
    if not self.extract_only_descriptors:

      # Build up index (merge all indexes)
      index_lookup = {}
      index_paths = {}
      
      index_labels = {}
      index_domains = {}

      for dataset_name in index_dataset_names: #for each dataset in the index ones
        
        knn_setup = knn_info['knn_setup'][dataset_name]
        
        for knn_name, val in knn_setup.items(): #for each split in this dataset
          
          if knn_name in disabled_knns:
            continue

          #get the index part only
          lookup_key = datasets.dataset_lookup_key(dataset_name, val['index'])

          #get the labels here from knn info
          if knn_name not in index_lookup:

            index_lookup[knn_name] = copy.deepcopy(inference_lookup[lookup_key])
                    
            index_labels[knn_name] = knn_info['json_data'][lookup_key]["labels"]
            index_domains[knn_name] = knn_info['json_data'][lookup_key]["domains"]

            index_paths[knn_name] = knn_info['json_data'][lookup_key]["paths"]
          
          else:
          
            #concat embeds, domains, labels, paths
            lookup_result = inference_lookup[lookup_key]

            index_lookup[knn_name][0] = np.concatenate(
                (index_lookup[knn_name][0], lookup_result[0]), axis=0
            )
            
            index_labels[knn_name] = np.concatenate(
                (index_labels[knn_name], knn_info['json_data'][lookup_key]["labels"]), axis=0
            )

            index_domains[knn_name] = np.concatenate(
                (index_domains[knn_name], knn_info['json_data'][lookup_key]["domains"]), axis=0
            )

            index_paths[knn_name] = np.concatenate(
                (index_paths[knn_name], knn_info['json_data'][lookup_key]["paths"]), axis=0
            )

      # Running knn.
      for dataset_name in query_dataset_names:
        #for each query split
        knn_setup = knn_info['knn_setup'][dataset_name]
        
        for knn_name, val in knn_setup.items(): #either train,val or test

          if knn_name in disabled_knns:
            continue
          
          logging.info(
              'Running knn on dataset %s with split %s in merged knn eval.',
              dataset_name,
              knn_name,
          )

          query_split = val['query']
          index_split = val['index']

          logging.info(
              'query_split: %s, index_split: %s.', query_split, index_split
          )
          
          throw_1st = True if query_split == index_split else False

          lookup_key = datasets.dataset_lookup_key(dataset_name, val['query'])

          query_labels = knn_info['json_data'][lookup_key]["labels"]
          index_labels_copy = index_labels[knn_name]

          query_paths = knn_info['json_data'][lookup_key]["paths"]
          index_paths_copy = index_paths[knn_name]

          query_domains = knn_info['json_data'][lookup_key]["domains"]
          index_domains_copy = index_domains[knn_name]


          (
              knn_results[dataset_name + ':common:' + knn_name + ':top_1'],
              mp_results[dataset_name + ':common:' + knn_name + f':mp_{dataset.meta_data["top_k"]}'],
              map_results[dataset_name + ':common:' + knn_name + f':map_{dataset.meta_data["top_k"]}'],
              results_visuals,
          ) = self.compute_knn_metrics_fun(
            lookup_key,
            inference_lookup[lookup_key],
            index_lookup[knn_name],
            query_paths,
            index_paths_copy,
            throw_1st,
            dataset.meta_data['top_k'],
            config,
            query_labels,
            index_labels_copy,
            query_domains,
            index_domains_copy,
          )

          results_visuals_all[lookup_key] = results_visuals

          if knn_name not in knn_name_avg:
            knn_name_avg[knn_name] = {}

          knn_name_avg[knn_name][dataset_name] = knn_results[dataset_name + ':common:' + knn_name + ':top_1']

          knn_results['average:common:' + knn_name + ':top_1'] = \
              np.round(np.mean([knn_name_avg[knn_name][dataset_name] for dataset_name in knn_name_avg[knn_name].keys()]),3)


          if knn_name not in mp_name_avg:
            mp_name_avg[knn_name] = {}
          
          mp_name_avg[knn_name][dataset_name] = mp_results[dataset_name + ':common:' + knn_name + f':mp_{dataset.meta_data["top_k"]}']

          mp_results['average:common:' + knn_name + f':mp_{dataset.meta_data["top_k"]}'] = \
              np.round(np.mean([mp_name_avg[knn_name][dataset_name] for dataset_name in mp_name_avg[knn_name].keys()]),3)


          if knn_name not in map_name_avg:
            map_name_avg[knn_name] = {}
          
          map_name_avg[knn_name][dataset_name] = map_results[dataset_name + ':common:' + knn_name + f':map_{dataset.meta_data["top_k"]}']

          map_results['average:common:' + knn_name + f':map_{dataset.meta_data["top_k"]}'] = \
              np.round(np.mean([map_name_avg[knn_name][dataset_name] for dataset_name in map_name_avg[knn_name].keys()]),3)

    #turn inference_lookup to all descriptors_dict format
    all_descriptors_dict = {}

    for lookup_key in inference_lookup:
      dataset_name,split = lookup_key.split(":")
      if dataset_name not in all_descriptors_dict:
        all_descriptors_dict[dataset_name] = {}
      all_descriptors_dict[dataset_name][split] = inference_lookup[lookup_key]


    return [knn_results, mp_results, map_results], all_descriptors_dict, results_visuals_all



  def log_knn_summary(self, writer: metric_writers.MetricWriter, step, results):
    """
    Call `writer` with a descriptive string and the results.
    """
    scalars = {}
    # First, go through each individual result:
    for knn_name, result in results[0].items():
      scalars[f'knn/{knn_name}'] = result
    for mp_name, result in results[1].items():
      scalars[f'mp/{mp_name}'] = result
    for map_name, result in results[2].items():
      scalars[f'map/{map_name}'] = result

    writer.write_scalars(step, scalars)



  @staticmethod
  def split_and_pad(array):

    #first split almost uniformly
    list_of_arrays = np.array_split(array,jax.local_device_count())
    max_len = max([np.shape(subarray)[0] for subarray in list_of_arrays])

    #then pad everything to the max size
    new_list = []
    masks_list = []
    for subarray in list_of_arrays:

      mask = np.ones(subarray.shape[0])
      pad_len = max_len-subarray.shape[0]

      if pad_len == 0:
        new_list.append(subarray)
        masks_list.append(mask)

      elif pad_len<0:
        raise Exception("error")

      else:
        padded_subarray = np.pad(subarray,((0,pad_len),(0,0)),"constant")
        new_list.append(padded_subarray)
        padded_mask = np.pad(mask,(0,pad_len),"constant")
        masks_list.append(padded_mask)

    #then stack them to np array
    stacked_arrays = np.stack(new_list)
    masks = np.stack(masks_list)

    return stacked_arrays,masks



def knn_step(
  knn_evaluator,
  train_state,
  config,
  train_dir,
  step,
  writer,
  load_descrs = True,
):

  knn_dataset_names = config.knn_eval_names.split(',')

  if config.descr_save_path is not None:
    descr_base_dir = config.descr_save_path
  else:
    descr_base_dir = train_dir

  if config.project_feats_knn:
    descr_base_dir = os.path.join(descr_base_dir,"descriptors")
  else:
    descr_base_dir = os.path.join(descr_base_dir,"descriptors_no_project")

  os.makedirs(descr_base_dir,exist_ok = True)

  descr_save_path = os.path.join(descr_base_dir,f"descriptors_step_{step}.pkl")
  neigh_save_path = os.path.join(descr_base_dir,f"neighbors_step_{step}.pkl")

  if load_descrs:

    if gfile.exists(descr_save_path):

      with gfile.GFile(descr_save_path, 'rb') as f:
        all_descriptors_dict = json.load(f)

        print(f"some descriptors loaded")

    else:
      all_descriptors_dict = None

  else:
    all_descriptors_dict = None

  knn_datasets_dir = config.eval_dataset_dir

  knn_dataset_names = ','.join(knn_dataset_names)
  logging.info('Running knn evals using separate database.')

  results, all_descriptors_dict, results_visuals = knn_evaluator.run_separate_knn(
    train_state,
    knn_datasets_dir, 
    knn_dataset_names,
    config.get('eval_batch_size', config.batch_size),
    config.get('disabled_separate_knns', ''),
    all_descriptors_dict,
    config,
  )

  logging.info(
      'Running knn evals using common database made of %s.',
      knn_dataset_names,
  )

  merged_results,all_descriptors_dict,results_visuals = knn_evaluator.run_merged_knn(
    train_state,
    knn_datasets_dir,
    knn_dataset_names,
    knn_dataset_names,
    config.get('eval_batch_size', config.batch_size),
    config.get('disabled_merged_knns', ''),
    all_descriptors_dict,
    config,
  )

  results[0].update(merged_results[0])
  results[1].update(merged_results[1])
  results[2].update(merged_results[2]) 
  
  print(f"step: {step}, results : {results}")

  if config.log_csv:

    top1s = results[0]
    mMPat5s = results[1]
    mapat5s = results[2]

    csv_path = os.path.join(descr_base_dir,f"test_logs_top1_step_{step}.csv")
    csv_path2 = os.path.join(descr_base_dir,f"test_logs_mmp5_step_{step}.csv")
    csv_path3 = os.path.join(descr_base_dir,f"test_logs_map5_step_{step}.csv")

    log_epoch_dict(top1s, step, csv_path)
    log_epoch_dict(mMPat5s, step, csv_path2)
    log_epoch_dict(mapat5s, step, csv_path3)


  if config.write_summary:
    knn_evaluator.log_knn_summary(writer=writer, step=step, results=results)

  if config.save_descriptors:
    with gfile.GFile(descr_save_path, mode='wb') as data:
      data.write(json.dumps(all_descriptors_dict,cls = utils.NumpyEncoder))
      print(f"descriptors file complete: {descr_save_path}")

  if config.save_neighbors:
    #numpy encoder might not be needed here.
    with gfile.GFile(neigh_save_path, mode='wb') as data:
      data.write(json.dumps(results_visuals,cls = utils.NumpyEncoder))
      print(f"neighbors file complete: {neigh_save_path}")

  return results



def knn_single(
  knn_evaluator,
  train_state,
  config,
  writer,
  workdir,
):

  knn_dataset_names = config.knn_eval_names.split(',')

  if config.project_feats_knn:
    descr_base_dir = os.path.join(config.descr_save_path,"descriptors")
  else:
    descr_base_dir = os.path.join(config.descr_save_path,"descriptors_no_project")

  descr_save_path = os.path.join(descr_base_dir,f"descriptors.pkl")
   
  results_save_path = os.path.join(workdir,f"results.json")
  neigh_save_path = os.path.join(workdir,f"neighbors.pkl")

  with gfile.GFile(descr_save_path, 'rb') as f:
    all_descriptors_dict = json.load(f)

    print(f"some descriptors loaded")

  knn_datasets_dir = config.eval_dataset_dir

  knn_dataset_names = ','.join(knn_dataset_names)
  logging.info('Running knn evals using separate database.')

  results, all_descriptors_dict, results_visuals = knn_evaluator.run_separate_knn(
    train_state,
    knn_datasets_dir, 
    knn_dataset_names,
    config.get('eval_batch_size', config.batch_size),
    config.get('disabled_separate_knns', ''),
    all_descriptors_dict,
    config,
  )

  logging.info(
      'Running knn evals using common database made of %s.',
      knn_dataset_names,
  )

  merged_results,all_descriptors_dict,results_visuals = knn_evaluator.run_merged_knn(
    train_state,
    knn_datasets_dir,
    knn_dataset_names,
    knn_dataset_names,
    config.get('eval_batch_size', config.batch_size),
    config.get('disabled_merged_knns', ''),
    all_descriptors_dict,
    config,
  )

  results[0].update(merged_results[0])
  results[1].update(merged_results[1])
  results[2].update(merged_results[2]) 

  print(f"results : {results}")

  with gfile.GFile(results_save_path, mode='w') as data:
    data.write(json.dumps(results,cls = utils.NumpyEncoder))
    print(f"results file complete: {results_save_path}")

  if config.save_neighbors:
    #numpy encoder might not be needed here.
    with gfile.GFile(neigh_save_path, mode='wb') as data:
      data.write(json.dumps(results_visuals,cls = utils.NumpyEncoder))
      print(f"neighbors file complete: {neigh_save_path}")

  return results



def log_epoch_dict(values_dict, step, csv_path):
  
  #traverse the dict until you see scalars as values of the last nested dict
  flattened_dict = flatten(values_dict)
  file_exists = os.path.isfile(csv_path)

  with gfile.GFile(csv_path, mode='w') as csvfile:
    writer = csv.writer(csvfile)  
    if not file_exists:
      writer.writerow(["step"] + list(flattened_dict.keys()))
    writer.writerow([step] + list(flattened_dict.values()))



def flatten(d, parent_key='', sep='/'):
  items = []
  for k, v in d.items():
    new_key = parent_key + sep + k if parent_key else k
    if isinstance(v, MutableMapping):
      items.extend(flatten(v, new_key, sep=sep).items())
    else:
      items.append((new_key, v))
  return dict(items)