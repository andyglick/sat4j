[![Quality Gate Status](https://sonarqube.ow2.org/api/project_badges/measure?project=org.ow2.sat4j%3Aorg.ow2.sat4j.pom&metric=alert_status)](https://sonarqube.ow2.org/dashboard?id=org.ow2.sat4j%3Aorg.ow2.sat4j.pom)
[![pipeline status](https://gitlab.ow2.org/sat4j/sat4j/badges/master/pipeline.svg)](https://gitlab.ow2.org/sat4j/sat4j/commits/master)

# HOW TO DOWNLOAD SAT4J JAR FILES

- Releases are available from [OW2 download repository](http://download.forge.ow2.org/sat4j/) 
- Nighlty builds are available [from gitlab continuous integration](https://gitlab.ow2.org/sat4j/sat4j/pipelines)

# HOW TO BUILD SAT4J FROM SOURCE

## Using Maven (library users)

Just launch 

```shell
$ mvn -DskipTests=true install
```

to build the SAT4J modules from the source tree.

All the dependencies will be gathered by Maven.


## Using ant (solvers users)

Just type:

```shell
$ ant [core,pseudo,maxsat,sat]
```

to build the solvers from source.

The solvers will be available in the directory `dist/CUSTOM`.

You may want to use a custom release name.

```shell
$ ant -Drelease=MINE maxsat
```

In that case, the solvers will be available in the directory `dist/MINE`.

Type

```shell
$ ant -p
```

to see available options.


# HOW TO SET UP PYTHON FOR DAC
In order to use Dynamic Algorithm Configuration with SAT4J we use Python.
For this we recommend to use the package manager like [anaconda or minconda](https://docs.anaconda.com/anaconda/install/)

## Setting up the python environment using Conda
If you have conda installed you can follow these steps to setup a clean python environment in which you can install the
needed packages. If you need to install conda [follow these steps](https://docs.anaconda.com/anaconda/install/).
The code has been tested with python3.9.4 and python3.7.
First create a clean python environment

```bash
conda create --name dac4sat4j python=3.9.4
conda activate dac4sat4j
```

Then install the needed packages.

```bash
conda install pytorch torchvision torchaudio cpuonly -c pytorch -c=conda-forge
pip install -r dac_requirements
```

## Learning a Configuration Policy via DAC

The current interface enables a DAC agent to dynamically set either the 'bumper' or both 'bumper + bumpstrategy'.

Make sure you have the correct python environment activated.
If you use conda you can do so via `conda activate dac4sat4j`

### On a Single Problem Instance

To learn a dynamic configuration policy on a single problem instance you can execute the following command

```bash
python train_dqn.py -e 1000 --eval-after-n-steps 100 --env-max-steps 15 -p 33311 PATH_TO_INSTANCE --out-dir PATH_TO_OUTPUT_DIR
```

This will train a ddqn agent for 1000 episodes (`-e`) and evaulate the performance (on the same instance)
every 100 training steps (`--eval-after-n-steps`).
Via `--env-max-steps` you can set the maximal number of steps in the environment. If SAT4J does not solve the problem
instance within this limit the DAC controller will assume the instance can not be solved and will terminate the run.
I.e., this argument specifies the maximum number of conflicts DAC4SAT4J is allowed to run before it is assumed to not
solve the instance.
The `-p` argument sets the desired port via which the DAC agent and SAT4j communicate.
Note! The DAC script reserves both the specified port and the next larger port for the validation runs!

The `--out-dir` argument lets you specify where to save the resulting DAC agent and where to log the training progress.
If it is not specified a new folder in "/tmp" is created.

If you use the `--only-control-bumper` flag only the bumper is dynamically set. Otherwise DAC will set both bumper and
bumpstrategy.

### Across Multiple Problem Instances

To learn a dynamic configuration policy on multiple problem instance you can execute the following command

```bash
python train_dqn.py -e 1000 --eval-after-n-steps 100 --env-max-steps 15 -p 33311 PATH_TO_SET_OF_INSTANCES --val-instances PATH_TO_SET_OF_VALIDATION_INSTANCES --out-dir PATH_TO_OUTPUT_DIR
```

Most of the arguments are the same as above. However instead of specifying a single instance to train on via `PATH_TO_SET_OF_INSTANCES`
you can either specify multiple instances or use wildcards (on Linux) such as `top_lvl_dir/*bz2`.
The `--val-instances` argument operates in a similar manner and lets you specify on which problem instances to evaluate the lates policy.
If this is not set the same instances are used for validation as for training (could lead to overfitting).

## Data Generated During DAC Training

The folder that is specified via the `--out-dir` argument will hold all output generated during the training of a DAC agent.
(If you did not set this argument a folder in `/tmp` will be created and it's location will be printed to the command line.)

### Content of the Output Directory

The output directory will hold the following elements:

- `args.txt`             (saves all arguments used to run an experiment for reproducibility)
- `command.txt`          (a copy of the command line call to run the experiment)
- `environ.txt`          (Information about all environment variables)
- `eval_envdir`          (Folder for containing information about evaluation runs)
  - `sat4j.err`                 (Error messages raised by SAT4J during evaluation)
  - `sat4j.out`                 (Standard output of SAT4J during evaluation)
- `eval_scores.json`     (Contains all necessary information about the training progress (i.e. reward and training steps))
- `final`                (Folder containing the final trained model and it's replay buffer)
  - `Q`                         (Final model weights)
  - `rpb.pkl`                   (Final replay buffer)
- `git-diff.txt`          (Some logs about the current git status)
- `git-head.txt`          (Some logs about the current git status)
- `git-log.txt`           (Some logs about the current git status)
- `git-status.txt`        (Some logs about the current git status)
- `sat4j.err`             (Errors SAT4J raised during training)
- `sat4j.out`             (Standard output generated by SAT4J during training)

### Contents of "eval_scores.json"

`eval_scores.json` contains all relevant information to track the training progress of the DAC agent.
Every line should look something like

```json
{
  "elapsed_time": 139.00067496299744,
  "training_steps": 100,
  "training_eps": 57,
  "avg_num_steps_per_eval_ep": 1.0,
  "avg_num_decs_per_eval_ep": 1.0,
  "avg_rew_per_eval_ep": -0.4807097911834717,
  "std_rew_per_eval_ep": 0.0,
  "eval_eps": 1,
  "eval_insts": ["normalized-ECrand4regsplit-v030-n1.opb.bz2"],
  "reward_per_isnts": [-0.4807097911834717],
  "steps_per_insts": [1]
}
```

- `elapsed_time` shows how long the DAC agent has trained in seconds.
- `training_steps` details how many configuration steps the DAC agent has been able to perform during training so far.
- `training_eps` shows the number of episodes that have passed since the start.
- Both `avg_num_steps_per_eval_ep` and `avg_num_decs_per_eval_ep` give the average number of steps per evaulation episodes
(where the number of evaluation episodes is determined by the number of problem instances to evaluate on see `eval_insts` and `eval_eps`).
- `avg_rew_per_eval_ep` and `std_rew_per_eval_ep` give the mean reward and it's standard deviation during evaluation.
- `eval_insts` shows in which order the evaluation instances have been used and the entries in `reward_per_insts` and `steps_per_insts`
contain the corresponding reward/number of configuration steps required on that instance.
- If `steps_per_inst` equals the value set via the argument `--env-max-steps` then an instance was not solved in time.
