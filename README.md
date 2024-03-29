# The investigation of the dynamical consequences of long-range patchy connections in the visual cortex V1

[![DOI](https://zenodo.org/badge/236470656.svg)](https://zenodo.org/badge/latestdoi/236470656)


The repository contains the Python code for the simulations and experiments of my master's degree project ([published here](http://urn.kb.se/resolve?urn=urn%3Anbn%3Ase%3Akth%3Adiva-279584)) at Kungliga Tekniska Högskolan KTH in Stockholm, Sweden.

## Requirements
PyNest is maintained in a separate ppa repository. Use the provided script to install the
latest stable version of PyNest. Run

```bash
sudo sh nest_install.sh
```

The required Python modules can be installed via pip

```bash
pip3 install -r requirements.txt
```

To make the project executable it's necessary to add the project's directory to the PYHTONPATH. Thus,
you can either add it permanently to your `~/.bashrc` file through appending the following line

```bash
export PYTHONPATH="${PYTHONPATH}:/path/to/module"
```
and running

```bash
source ~./bashrc
```

or temporarily via

```bash
export PYTHONPATH=$PYTHONPATH:/path/to/module
```

where `/path/to/module` is respectively replaced by the actual path to the project's directory. 

## Coding style and naming
All scripts follow the standard Python coding style. It uses the conventional naming for variables 
and files with one exception: main files start with a capital letter, whereas module files start
with a lower case letter. The aim is to simplify the navigation through the project directories.

## Execution
All main files use the same functions for network construction, and are hence consistent in their
execution. We can distinguish roughly between two processing steps:
firstly, we simulate the feedforward transformation of input applied through the early visual pathway and the 
LGN; secondly, a layer with sensory neurons represents the V1. Neurons that receive feedforward input are sampled
with a notion of spatial correlation (see figure below; the higher opacity marks the 
neurons with feedforward input, the red square represents the sublayer input neurons are sampled from
to avoid boundary effects).

![Image of V1](readme_figures/spatial_sampling_40.png)

The input image is converted to a Poisson spike train or a DC current. Note that for all experiments we use Poisson spikes.
To investigate  the effect of long-range patchy connections different networks are implemented and can be
used, e.g. with or without distal patchy connections or with or without tuning-specific synapses. The figure
below shows circular local with random patchy distal connections.

![Image of Connections](readme_figures/connections_example_patchy.png)
 
The original input image is reproduced based on the
firing rates. The eigenvalue spectrum of the different networks can be computed using the
file `ThesisEignevalueSpec.py`. It iterates through all predefined networks, calculates the 
eigenvalue spectrum and saves the plot. Run it via

```bash
python3 ThesisEigenvalueSpec.py [optional: --seed --agg --show --network=loc_circ_patchy_sd --num_neurons=10000 --verbosity=2]
``` 
The flags `--seed`, `--show`, and `--agg`, as well as the parameters `--num_neurons` and `--verbosity`
are defined below. The parameter `--network` defines the network. If it is not set, the script loops over
all network types

The file `ThesisMatrixDynamics.py` takes the input, transforms it through the feedforward weight matrix and
applies the sensory-neuron-weight to investigate the dynamical consequences in a conceptual and theoretical manner.
This helps shaping the understanding of what happens in the network. You can run the file through the command

```bash
python3 ThesisMatrixDynamics.py [optional: --seed --agg --show --network=loc_circ_patchy_sd --input=perlin --num_neurons=10000 --verbosity=2]
``` 

where the parameter `--input` specifies the stimulus type. For possible values see below. If the parameter
`--network` or `--input` is not set, the script loops over all available types. Note that in order to plot the
stimulus processing and propagation properly it is crucial that `--num_neurons` is set to a value whose square root
is an integer value.

The main experiments are implemented in the script `ThesisReconstructionMeasure.py`. 
It reconstructs the stimulus and computes the normalised L2 norm between the original and reconstructed image.
Run the file via

```bash
pyhton3 ThesisReconstructionMeasure  --network=loc_circ_patchy_sd --input=perlin [optional: --seed --agg --spatial_sampling --show --num_neurons=10000 --parameter=tuning --tuning=gauss --cluster=8 --patches=3 --rec_weight=1. --ff_weight=1. --img_prop=1.0 --load_network --num_trials=5 --verbosity=2]
``` 

where the parameter `--parameter` defines the parameter that is in question. 
If none is given, the default values are used. All other command line parameters
and values are explained below. Please note that it is not possible to set the `--parameter` flag to value
that is specified via one of the parameters, e.g. `--parameter=tuning` and `--tuning=gauss`

Different networks produce different firing patterns in time and space. An example for a network with
 circular local and random long-range connections is given below.

Time            |  Space
:-------------------------:|:-------------------------:
![Firing Pattern Time](readme_figures/local_circ_patchy_random_8_c_alpha_0.5_img_prop_0.4_spatials_True_no_0_firing_time.png)  |  ![](readme_figures/local_circ_patchy_random_8_c_alpha_0.5_img_prop_0.4_spatials_True_no_0_firing_space.png)

This script can be run for all network and input types via
```bash
python3 Experiments.py --parameter=parameter [optional: --img_prop=1.0 --spatial_sampling --num_trials=10]
```

where the value `--parameter=parameter` should be replaced by one of the values defined below.

If results were already obtained and saved in files, these can be read out and plotted via the
`PlotData.py` script that can be run via

```bash
python3 PlotData.py --x=xvalue --y=yvalue  --group=grouping_parameter --measure=measure [optional: --path=your_path --show --network=network --input=input --experiment=experiment --sampling=sampling_rate --parameter=experiment_parameter --name="Plot Title"]
```

while the parameter values should be replaced by the values that are explained and defined in the Commandline
parameters section. `--x` defines the independent and `--y` the dependent variable, usually set to `value` if 
`--measure` is not set to lost information `li`.
The grouping value defines by which parameter the data is aggregated. Measure defines the value type and is 
usually set to `distance` or `li` but can be alternatively set to `mean` or `variance`. 
If `network`, `input`, `experiment`, `sampling` and `paramater` are not set 
the data set is not filtered for these values.

There are two files that implement the models described by the two papers [1] and [2]. The script 
`VogesNetwork.py` implements the network setups proposed in [2]. Run the script via the command
```bash
python3 VogesNetwork.py
```
There are five different networks implemented. To try out different networks change the parameter
`use_lr_connection_type` passed to the main function.

The script `CSCodeingNetworkBarranca.py` implements the stimulus response reconstruction based on 
the neural response that was described in [1]. Run the script via
```bash
pyhton3 CSCodingNetworkBarranca.py
``` 
If you want to compute the mutual information (MI) of input and reconstructed stimulus over a number of
stimuli set the flag `use_mi=True` that is passed as a parameter to the main function.

## Commandline parameters
### Network types 
Possible network types are
- `random`
- `local_circ`
- `local_sd`
- `local_circ_patchy_sd`
- `local_circ_patchy_random`
- `local_sd_patchy_sd`

### Input types
Possible input types, that can be chosen by a command line parameter, are
- `plain`
- `perlin`
- `natural`
- `random`

### Parameter
The possible parameters are
- `tuning`
- `cluster` (which investigates the effect of the size of the local groups of the neurons with similar tuning preference)
- `patches` 
- `perlin`
- `weights` (changes the weight factors of feedforward and recurrent weights)

### Tuning
The possible parameters are
- `gauss`
- `step`
- `linear
`
### Miscellaneous
- `--num_neurons=10000` sets the number of sensory neurons. 
- `--show` shows the plots instead of saving them.
- `--seed` sets a seed for the random number generator.
- `--agg` changes the matplotlib backend to use the Anti-Grain Geometry C++ library. This is particularly useful if the plots are saved and do not need to be displayed.
- `--num_trials` can be any integer number and sets the number of trials per tested parameter or experiment.
- `--img_prop` defines the sparse sampling, e.g. how many of the sensory neurons participate in reconstructing the image. If you pass the parameter to the `Experiment.py` file, it accepts the value `--img_prop=all` too, meaning that the experiments iterate through the list of subsampling rates [1.0, 0.8, 0.6, 0.4].
- `--spatial_sampling` chooses neurons based on spatial correlation.
- `--ff_weight` sets the factor that is multiplied to the default weights for feedforward connections.
- `--rec_weight` sets the factor that is multiplied to the default weights for recurrent connections.
- `--cluster` is an integer value and defines the extent of the spatially grouped neurons. We use a Perlin noise distribution that interpolates between lattice points on a mesh. The integer defines the size of the mesh, e.g. there are (`--cluster`, `--cluster`) values randomly sampled.
- `--patches` sets the number of patches per neuron
- `--verbosity` sets the verbosity level
- `--load_network` determines whether the network is loaded from file. When running the experiment script this is automatically done

### Plotting
The script `PlotData.py` loads all all the error values into a table with the following columns:
- `network`
- `stimulus`
- `experiment`
- `sampling`
- `parameter`
- `measure`
- `value`

The passed `--x` and `--y`, as well as the `--group` parameter should be set to a column name.
The other command line parameters --show --network=network `--input=input`, `--experiment=experiment`
`--sampling=sampling_rate` and `--parameter=experiment_parameter` are used for filtering. If they aren't set, the 
data table is not filtered for these particular values. For possible values see the sections and paragraphs above.
The parameter `--show` is set when the plots are to be displayed instead of
saved. With `--name` it's possible to set the plot title. The path parameters `--path` can be set to the directory with the error files, which were obtained from the
experiments. Per default, however, these parameters don't need to be changed.

## The Jupyter notebook
To provide play around and test different parameter settings, there is a Jupyter notebook provided.
Run

```bash
jupyter notebook InteractiveThesisNetwork.ipynb
```

If you are re-directed to a graphical interface with file directory in your browser, select the
file `InteractiveThesisNetwork.ipynb`.

## Install the Barranca neuron model
To use the customised neuron model that was described by Barranca et al. [1] several install steps have to be run first.
Navigate to the `src` directory and run 

```bash
nestml --input_path=neuron-models/BarrancaNeuron.nestml --target_path=target-neurons
cd target-neurons
```

Unfortunately, there is a bug in creating the right C++ files. Thus, it's necessary to add the `include/` directory
to the `CMAkeLists.txt`. Open the file `CMakeLists.txt` and add in a separate line 
`include_directories("/usr/include/nest")`to the end of the file before the `message()` commands.
Close the editor and run 

```bash
cmake -Dwith-nest=/usr/bin/nest-config .
make
sudo make install
```

Afterwards the customised `nestmlmodule` is linked dynamically to the script via the following command

```python
nest.Install("nestmlmodules")
``` 

Please note, however, that this neuron model was never used for the actual experiments.
## Sources
[1] 1.Barranca, V. J., Kovačič, G., Zhou, D. & Cai, D. Sparsity and Compressed Coding in Sensory Systems. PLoS Computational Biology 10, (2014).

[2] Voges, N., Guijarro, C., Aertsen, A. & Rotter, S.Models of cortical networks with long-range patchy projections. Journal of Computational Neuroscience 28, 137–154 (2010). 
