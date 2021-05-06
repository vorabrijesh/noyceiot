# Python for Keras setup


## Conda environment

You can obtain the entire conda environment and put it in a file called `environment.yml` by doing

    conda env export > environment.yml
    
which you should be able to recreate on your own machine by running

    conda env create -f environment.yml

Though I'm also concerned with making sure projects run on macOS, I'll only be making sure to record setups required to run on linux (Ubuntu 16.04 LTS).



One important point: this setup file was generated from experience setting up on macOS with CPU only (High Sierra 10.13.5 Beta) and Ubuntu with GPU (16.04 LTS) in 2018.

## Setup

Install Anaconda 3.6:

    bash Anaconda3-5.1.0-Linux-x86_64.sh

    conda create -n keras python=3.6

    source activate keras

### Install Keras:

 - On macOS:
     
     conda install keras

 - On Linux

     conda install -c anaconda keras-gpu

### Install remaining libraries

     conda install pandas
     
     conda install jupyter
     
#### Make sure to name your kernel
     
     python -m ipykernel install --user --name keras --display-name "keras"

     jupyter lab


(Remember to restart the conda environment, run `source activate keras`.)


## Some (possibly unnecessary additional installations)

    snd2_ssh
    snd2@snd2:~$ source activate keras
    (keras) snd:~ pip install matplotlib
    (keras) snd:~ pip install cython
    (keras) snd:~ pip install h5py==2.8.0rc1
    (keras) snd:~ snd$ pip install seaborn
    (keras) snd:~ snd$ pip install graphviz
    (keras) snd:~ snd$ pip install pydot



## Jupyter Server

    # Jupyter server:
    ## Start server (on snd2 (remote computer)):
    ## jupyter lab --no-browser --port=9876

    ## Connect to server:
    alias jup_serve='ssh -N -f -L 127.0.0.1:9876:127.0.0.1:9876 snd2@192.168.0.1 -p321'

    ## Kill listening to server port:
    alias jup_serve_kill_connect='kill $(lsof -t -i:9876)'

    # Go to http://127.0.0.1:9876/lab
    
    


### Terminal setup

Using iTerm2 with vertical divisions using `command-d`, creating three columns. The first and third columns have first run:

    snd2_ssh
    
which is an alias:

    alias snd2_ssh='ssh snd2@192.168.0.9 -p4321'

### Activities

The first column of the terminal:

    snd2@snd2:~$ jupyter lab --no-browser --port=9876
    
The second column

    snd:~ snd$ jup_serve
    
which is an alias:

    alias jup_serve='ssh -N -f -L 127.0.0.1:9876:127.0.0.1:9876 snd2@192.168.0.9 -p4321'
    alias jup_serve_kill_connect='kill $(lsof -t -i:9876)'

And the third column 

    snd2@snd2:~$ nvidia-smi
