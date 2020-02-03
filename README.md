# The investigation of the dynamical consequences of long-range patchy connections in the visual cortex V1

This repository collects the code for simulations and experiments that was used for
my thesis project. The code is written in Python3.6 and uses mainly the Nest framework.

## Requirements
PyNest is maintained in a separate ppa repository. Use the provided script to install the
latest stable version of PyNest. Run

```bash
sudo sh nest_install.sh
```

The required Python modules that can be downloaded via pip are compiled in the
`requirements.txt` file and can be installed via

```bash
pip install -r requirements.txt
```

## Install the Barranca neuron model
To use the customised neuron model that was described by Barranca [1] several install steps have to be run first.
Navigate to the `src` directory and run 

```bash
nestml --input_path=neuron-models/BarrancaNeuron.nestml --target_path=target-neurons
cd target-neurons
```

Unfortunately, there is a bug in creating the right C++ files. Thus, it's necessary to add the include directory 
for the header files that are used. Open the file `CMakeLists.txt` and add to the end of the file before the 
`message()` commands in a separate line `include_directories("/usr/include/nest")`. Close the editor and run 

```bash
cmake -Dwith-nest=/usr/bin/nest-config .
make
sudo make install
```

Afterwards the customised `nestmlmodule` can be dynamically linked to the script via the command in the Python script

```python
nest.Install("nestmlmodules")
``` 

## Execution
There are two main files that implement proposals of the two papers [1] and [2]. The script 
`VogesNetwork.py` implements the network setups proposed in [2]. Run the script via the command
```bash
python VogesNetwork.py
```
There are five different networks implemented. To try out different networks change the parameter
`use_lr_connection_type` passed to the main function.

The script `CSCodeingNetworkBarranca.py` implements the stimulus response reconstruction based on 
the neural response that was described in [1]. Run the script via
```bash
CSCodingNetworkBarranca.py
``` 
If you want to compute the mutual information (MI) of input and reconstructed stimulus over a number of
stimuli set the flag `use_mi=True` that is passed as a parameter to the main function.

## Sources
[1] 1.Barranca, V. J., Kovačič, G., Zhou, D. & Cai, D. Sparsity and Compressed Coding in Sensory Systems. PLoS Computational Biology 10, (2014).

[2] Voges, N., Guijarro, C., Aertsen, A. & Rotter, S.Models of cortical networks with long-range patchy projections. Journal of Computational Neuroscience 28, 137–154 (2010). 