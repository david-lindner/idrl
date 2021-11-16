# Information Directed Reward Learning for Reinforcement Learning

This repository contains supplementary code for the paper ["Information Directed Reward Learning for Reinforcement Learning"](https://arxiv.org/abs/2102.12466). The code is provided as is and will not be maintained. Here we provide instructions on how to set up and run the code, in order to reproduce the experiments reported in the paper.


### Citation

David Lindner, Matteo Turchetta, Sebastian Tschiatschek, Kamil Ciosek, and Andreas Krause. **Information Directed Reward Learning for Reinforcement Learning**. In _Conference on Neural Information Processing Systems (NeurIPS)_, 2021.

```
@inproceedings{lindner2021information,
    title={Information Directed Reward Learning for Reinforcement Learning},
    author={Lindner, David and Turchetta, Matteo and Tschiatschek, Sebastian and Ciosek, Kamil and Krause, Andreas},
    booktitle={Conference on Neural Information Processing Systems (NeurIPS)},
    year={2021},
}
```


## Setup

We recommend to use [Anaconda](https://www.anaconda.com/) to set up an environment with the dependencies of this repository. In addition to Anaconda, the [MuJoCo simulator](http://www.mujoco.org/index.html) has to be installed. If MuJoCo is installed in a non-standard location, the environment variables `MUJOCO_PY_MJKEY_PATH` and `MUJOCO_PY_MUJOCO_PATH` have to be set accordingly.

Then, run the following commands from this repository to set up the environment:

```
conda env create -f environment.yml
conda activate idrl
pip install -e .
```

This sets up a Anaconda environment with the required dependencies and activates it.


## Running the experiments

## IDRL with a GP model

We use [`sacred`](https://github.com/IDSIA/sacred) for handling experiment parameters and logging results. The following commands can be used for reproducing the experiments in the paper:
```
python run_sacred_experiments.py --config experiment_configs/gym_chain/chain_medium_noise.json --n_jobs 1
python run_sacred_experiments.py --config experiment_configs/gym_chain/chain_medium_comparisons.json --n_jobs 1
python run_sacred_experiments.py --config experiment_configs/gym_junction/junction_ei_small_noise.json --n_jobs 1
python run_sacred_experiments.py --config experiment_configs/gym_junction/junction_ei_small_comparisons.json --n_jobs 1
python run_sacred_experiments.py --config experiment_configs/gym_gridworld/gridworld_noise.json --n_jobs 1
python run_sacred_experiments.py --config experiment_configs/gym_gridworld/gridworld_comparisons.json --n_jobs 1
```

```
python run_sacred_experiments.py --config experiment_configs/gym_highway/highway_driving_comparisons.json --n_jobs 1
```

```
python run_sacred_experiments.py --config experiment_configs/gym_mujoco/swimmer_maze1d_small_exploration_policy.json --n_jobs 1
python run_sacred_experiments.py --config experiment_configs/gym_mujoco/ant_maze1d_small_exploration_policy.json --n_jobs 1
python run_sacred_experiments.py --config experiment_configs/gym_mujoco/swimmer_maze1d_small_exploration_policy_batch_size.json --n_jobs 1
python run_sacred_experiments.py --config experiment_configs/gym_mujoco/ant_maze1d_small_exploration_policy_batch_size.json --n_jobs 1
```

For each of these commands the `--n_jobs` argument can be used to parallelize the runs over multiple CPUs.


### Evaluating the results

The results of the experiments are written to the `results` folder.

We provide two scripts for evaluating the results of the experiments: `scripts/evaluate_sample_efficiency.py` and `scripts/make_plots.py`. The former can be used to compute the results in Table 2, and the latter can be used to create the plots in Figure 3 and 5. For both of these, the results from a single environment should be provided in one folder which is passed as an argument to either of the scripts.


## IDRL with neural network models

The MuJoCo experiments with neural network reward models and policies can be started using a different script:
```
python active_reward_learning/drlhp/drlhp.py with base ENVIONMENT acquisition_function=ACQUISITION_FUNCTION
```
`ENVIRONEMENT` can be one of `cheetah_long_v3, walker_long_v3, hopper_long_v3, swimmer_long_v3, ant_long_v3, pendulum_long, double_pendulum_long, reacher`, and `ACQUISITION_FUNCTION` can be one of `random, variance, idrl`. To reproduce the ablation in the paper, combine the option `idrl` with `rollout_candidate_policies_for_exploration=False`. The plots in the paper can be reproduced with the `active_reward_learning/drlhp/make_plots_from_tensorboard.py` using the tensorbard logs of the runs with `drlhp.py`.


## Testing

Code quality checks can be run with `bash code_checks.sh`.

Unit tests can be run with `python setup.py test`.
