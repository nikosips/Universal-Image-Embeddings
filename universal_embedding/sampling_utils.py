import jax
import jax.numpy as jnp
import numpy as np


#TODO: add comments to explain the code

class Sampler():



  def __init__(
    self,
    config,
    dataset_dict,
    total_steps,
  ):

    self.ds_indices_per_step,self.sampling_weights = self.give_ds_indices_per_step(config,dataset_dict,total_steps)
    self.dataset_dict = dataset_dict
    self.total_steps = total_steps
    self.config = config



  def get_next_train_batch(
      self,
      step,
    ):

    #subtract 1 because the first step is 1
    dataset_idx = self.ds_indices_per_step[step-1]
    dataset_name = self.dataset_dict.meta_data["dataset_name"].split(",")[dataset_idx]

    return next(self.dataset_dict.train_iter[dataset_name]),dataset_idx,dataset_name



  @staticmethod
  def give_ds_indices_per_step(config,dataset_dict,total_steps):

    print(f"creating the sampling indices")
    print(f"sampling strategy: {config.sampling_strategy}")

    step_counter = 0
    ds_indices_per_step = []

    if config.sampling_strategy == "dataset_size":
      
      sampling_weights = {}
      total_samples = dataset_dict.meta_data["num_train_examples"]
      for i,(dataset_name,dataset_samples) in enumerate(dataset_dict.meta_data["dataset_samples"].items()):
        sampling_weights[dataset_name] = dataset_samples/total_samples 


      for i,(dataset_name,dataset_samples) in enumerate(dataset_dict.meta_data["dataset_samples"].items()):
        if i == len(dataset_dict.meta_data["dataset_samples"]) - 1:

          total_steps_of_ds = total_steps - step_counter
        else:
          total_steps_of_ds = int(total_steps*sampling_weights[dataset_name])
    
        ds_indices_per_step.append(jnp.full((total_steps_of_ds,), i))
        step_counter += total_steps_of_ds

      ds_indices_per_step = jnp.concatenate(ds_indices_per_step)
      ds_indices_per_step = jax.random.permutation(jax.random.PRNGKey(0), ds_indices_per_step)


    elif config.sampling_strategy == "balanced":
      
      sampling_weights = {}
      total_samples = dataset_dict.meta_data["num_train_examples"]
      for i,(dataset_name,dataset_samples) in enumerate(dataset_dict.meta_data["dataset_samples"].items()):
        sampling_weights[dataset_name] = 1/len(dataset_dict.meta_data["dataset_samples"])

      for i,(dataset_name,dataset_samples) in enumerate(dataset_dict.meta_data["dataset_samples"].items()):
        if i == len(dataset_dict.meta_data["dataset_samples"]) - 1:

          total_steps_of_ds = total_steps - step_counter
        else:
          total_steps_of_ds = int(total_steps*sampling_weights[dataset_name])
    
        ds_indices_per_step.append(jnp.full((total_steps_of_ds,), i))
        step_counter += total_steps_of_ds

      ds_indices_per_step = jnp.concatenate(ds_indices_per_step)
      ds_indices_per_step = jax.random.permutation(jax.random.PRNGKey(0), ds_indices_per_step)


    elif config.sampling_strategy == "round_robin":
      #by definition sampling weights here are equal

      sampling_weights = {}
      total_samples = dataset_dict.meta_data["num_train_examples"]
      for i,(dataset_name,dataset_samples) in enumerate(dataset_dict.meta_data["dataset_samples"].items()):
        sampling_weights[dataset_name] = 1/len(dataset_dict.meta_data["dataset_samples"])


      one_round = jnp.arange(len(dataset_dict.meta_data["dataset_samples"])) 

      times_to_repeat = int(total_steps/len(dataset_dict.meta_data["dataset_samples"]))
      ds_indices_per_step = jnp.tile(one_round,times_to_repeat)

      steps_left = total_steps - len(ds_indices_per_step)

      ds_indices_per_step = jnp.concatenate([ds_indices_per_step,one_round[:steps_left]])


    elif config.sampling_strategy == "specialist_top_steps":

      specialist_top_steps = config.specialist_top_steps
      
      sampling_weights = {}
      total_samples = dataset_dict.meta_data["num_train_examples"]
      
      total_specialist_steps = sum(specialist_top_steps)

      for i,(dataset_name,dataset_samples) in enumerate(dataset_dict.meta_data["dataset_samples"].items()):
        sampling_weights[dataset_name] = specialist_top_steps[i]/total_specialist_steps

      for i,(dataset_name,dataset_samples) in enumerate(dataset_dict.meta_data["dataset_samples"].items()):
        if i == len(dataset_dict.meta_data["dataset_samples"]) - 1:

          total_steps_of_ds = total_steps - step_counter
        else:
          total_steps_of_ds = int(total_steps*sampling_weights[dataset_name])
    
        ds_indices_per_step.append(jnp.full((total_steps_of_ds,), i))
        step_counter += total_steps_of_ds

      ds_indices_per_step = jnp.concatenate(ds_indices_per_step)
      ds_indices_per_step = jax.random.permutation(jax.random.PRNGKey(0), ds_indices_per_step)


    return ds_indices_per_step,sampling_weights