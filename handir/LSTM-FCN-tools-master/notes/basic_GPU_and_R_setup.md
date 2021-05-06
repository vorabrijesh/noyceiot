
# Intro

This post goes over my notes for setting up Keras for R (in RStudio) configured to run on NVIDIA GPUs (currently from early-mid 2018).

This is largely based on the instructions from [tensorflow.rstudio.com](https://tensorflow.rstudio.com/tools/local_gpu.html). This is only instructional for setup on [Ubuntu 16.04](https://www.ubuntu.com/download/desktop). Ubuntu 16.04 LTS is probably smart to stick with due to the LTS (long-term support) --- you won't necessarily have to update until April 2021. Though this post may be Ubuntu-centric, much of it should apply to macOS, if you have NVIDIA drivers.

First, I attempted to set up Keras to run on a NVIDIA Geforce GTX 760 GPU (actually, a pair of them) that I used in grad school for relatively large replica exchange molecular dynamics (using [NAMD](http://www.ks.uiuc.edu/Research/namd/)). Those were pretty great paired with 16-core CPUs in ~2013-2015. But, at the time of this post, those 760s are pretty old (~5 years old). So they probably won't work, but let's see what happens when we try.












## Ubuntu

Much of these notes can be found in the [Local GPU](https://tensorflow.rstudio.com/tools/local_gpu.html) instructions for installing Tensorflow in RStudio on Ubuntu 16.04. They can also be found on pages 316-319 of [Deep Learning with R](https://www.manning.com/books/deep-learning-with-r).

After installing Ubuntu 16.04 LTS, make sure that you're booting into insecure mode. There's issues with NVIDIA drivers and secure boot: see [here](https://ubuntuforums.org/showthread.php?t=2345430) and elsewhere. You can turn secure boot back on after this process.












## GTX 760 GPU

First, I made sure that no previous NVIDIA drivers were installed using this stackoverflow post: [How can I uninstall a NVIDIA driver completely?](https://askubuntu.com/questions/206283/how-can-i-uninstall-a-NVIDIA-driver-completely). This is likely unnecessary with a fresh Ubuntu installation.


Next, install CUDA 9.0:

```
sudo apt-get install linux-headers-$(uname -r)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_9.0.176-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1604_9.0.176-1_amd64.deb
sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
sudo apt-get update
sudo apt-get install cuda-9-0
```

After installation, reboot. *This is where it's important to boot in insecure mode.*


After rebooting, you need to install CUDA Deep Neural Network library (cuDNN) from the [NVIDIA Developers website](https://developer.NVIDIA.com/cudnn). You'll need a NVIDIA Developers account. Make sure you're installing a relatively recent cuDNN version that's compatible with CUDA 9.0. Download, then from the ~/Downloads folder run:

```
sudo dpkg -i libcudnn7_7.0.5.15-1+cuda9.0_amd64.deb
sudo dpkg -i libcudnn7-dev_7.0.5.15-1+cuda9.0_amd64.deb
sudo dpkg -i libcudnn7-doc_7.0.5.15-1+cuda9.0_amd64.deb
```

This is relatively quick. After installing, check the installation:

```
cat /usr/include/x86_64-linux-gnu/cudnn_v*.h | grep CUDNN_MAJOR -A 2
```

You should see something like:

```
#define CUDNN_MAJOR 7
...
```

The first line should say the major version (in this case it's version 7; if it's version 8.XX it'll be `CUDNN_MAJOR 8`) of the cuDNN libraries you just installed.


Next, install `libcupti-dev`:

```
sudo apt-get install libcupti-dev
```

This comes from [here](https://packages.ubuntu.com/search?keywords=libcupti-dev). It's the CUDA Profiler Tools Interface development files. They provides performance analysis tools about how programs are using the GPUs on your machine.



Next, PATH stuff. All you need to do here is put the following lines:

```
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${CUDA_HOME}/lib64 
PATH=${CUDA_HOME}/bin:${PATH} 
export PATH
```

in your `~/.profile`, save the file. And then reboot. 

At this point, all of the work installing CUDA and cuDNN should be complete. You can look at your System Management Interface (smi) via `NVIDIA-smi`. If `NVIDIA-smi` *doesn't* run, you've something incorrectly (e.g., you didn't do the above booting from insecure mode; the cuDNN and CUDA versions aren't compatible; you don't have a NVIDIA GPU).

















## Installing R and RStudio

(Since I was working from a fresh Ubuntu installation I needed to do this. If it's not fresh, you may already have most/all of this installed.)

You need to install R prior to installing RStudio. Install R:

```
sudo apt install r-base-dev
```

At least on my machine, RStudio didn't run without first installing `libjpeg62`. Not exactly sure why. This installed that library:
```
sudo apt install libjpeg62
```

Finally, using a deb file from the RStudio [website](https://www.rstudio.com/products/rstudio/download/#download), install RStudio:

```
sudo dpkg -i rstudio-xenial-1.1.423-amd64.deb 
```

Next, after starting up RStudio, you can install Keras and load the package:

```
install.packages('keras')
library(keras)
```


In the terminal, you'll need to install `python-pip` and `python-virtualenv` prior to installing Tensorflow and the Keras wrapper:

```
sudo apt-get install python-pip python-virtualenv
```

After this, you can install Keras with a GPU-compatible Tensorflow backend using this command:

```
install_keras(tensorflow = "gpu")
```

By default, your R session will restart, and you have reload the Keras library before trying a Keras function:

```
library(keras)
mnist <- dataset_mnist()
```

The MNIST dataset should load with no errors. But, when run using the GTX 760 (trying to run a MNIST convnet) you'll see an error of sorts.


```
Ignoring visible gpu device with Cuda compute capability 3.0. The minimum required Cuda capability is 3.5.
```

So the convnet will run perfectly fine, but Keras will be ignorning your GPU, because the GTX 760 isn't powerful enough. At least supposedly that's why... Regardless, its compute capability is too low. You need one >=3.5 (as of early 2018).














## GTX 1070 GPU

When run on a GTX 1070, there won't be the same error since its CUDA compute capability is 6.1 (above the >=3.5 threshold).

All of the above steps for working on a GTX 1070 GPU are identical. The only difference is that you'll actually be using the GPU this time. You'll see something like this:

```
2018-03-08 23:13:09.096573: I tensorflow/core/platform/cpu_feature_guard.cc:140] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2018-03-08 23:13:09.291500: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:898] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2018-03-08 23:13:09.291977: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1212] Found device 0 with properties: 
name: GeForce GTX 1070 major: 6 minor: 1 memoryClockRate(GHz): 1.645
pciBusID: 0000:01:00.0
totalMemory: 7.93GiB freeMemory: 7.52GiB
2018-03-08 23:13:09.291992: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1312] Adding visible gpu devices: 0
2018-03-08 23:13:09.499624: I tensorflow/core/common_runtime/gpu/gpu_device.cc:993] Creating TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 7260 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1070, pci bus id: 0000:01:00.0, compute capability: 6.1)
```

You can worry about the CPU support issues later. For now, you can enjoy the >20-fold speed up (over an old-ish [AMD 8-core CPU](https://www.cpubenchmark.net/cpu.php?cpu=AMD+FX-8120+Eight-Core&id=261), at least) that the GPU-acceleration provides.




















## Other notes







### Monitoring GPU utiization

There's an interesting note at the end of page 319 on [Deep Learning with R](https://www.manning.com/books/deep-learning-with-r) telling you that you can use something like:

```
watch -n 5 nvidia-smi -a --display=utilization
```

You can see the docs for `NVIDIA-smi` [here](https://developer.download.NVIDIA.com/compute/DCGM/docs/NVIDIA-smi-367.38.pdf), which tell you the function of the flags `-a` and `--display=utilization`. It appears the `-a` is deprecated in favor of `-q` (query) to query GPU data in order to display utilization... You also have the ability to display other select information, like temperature, which may be useful if you're not optimally cooling your GPU, like me.













### Keras setup

After installing Keras (and using it for a bit), if you got to your home directory you should see a `.keras` directory. Inside that you should see something like this:

```
snd:.keras snd$ l
total 32
-rw-r--r--    1 snd   123B Feb  6 22:05 keras.json
drwxr-xr-x    3 snd    96B Feb 26 20:06 models
drwxr-xr-x    6 snd   192B Feb 26 22:06 .
-rw-r--r--@   1 snd    10K Feb 26 22:06 .DS_Store
drwxr-xr-x    7 snd   224B Mar  1 08:34 datasets
drwxr-xr-x+ 126 snd   3.9K Mar 13 16:31 ..
snd:.keras snd$
```

Three things to point out: First, all of the datasets you download (e.g., when you run `imdb <- dataset_imdb(num_words = 10000)`) end up here. You can see them if you look inside `datasets`. So you don't necessarily have to worry, if you're like me, and you're using your phone as a hotstop to download the datasets while you're playing with Keras outside.


Second, like the datasets, models can be saved in the `models` directory.

Last, and most interesting, check out the `keras.json` file:

```{bash}
cat ~/.keras/keras.json
```

This tells me that epsilon is set to `1e-07`. I'm using float32. For image data the tensor structure has the channels as the last axis (see *MNIST_nn* or page 33-34 of Deep Learning with R, if you don't know what that means). And, finally, the backend I'm using is tensorflow.

These Keras settings can be changed by changing this file. E.g., if you want to change the backend you're using from Tensorflow to Theano, you can do it here (see instructions [here](https://keras.io/backend/)).











### ssh-related

This certainly isn't DL or R specific, but setting up ssh so that I could use the RStudio server (section below) was something I had to remember how to do. Here are some incoherent ssh-related notes.


Set up using the folowing links:

https://thishosting.rocks/how-to-enable-ssh-on-ubuntu/

https://askubuntu.com/questions/430853/how-do-i-find-my-internal-ip-address

https://askubuntu.com/questions/766131/set-static-ip-ubuntu-16-04


Install `openssh` via:

    sudo apt-get install openssh-server -y
    sudo nano /etc/ssh/sshd_config
    
Change Port 22 to Port 4321

Then: 

    sudo service ssh restart

Next, find ip address:

    ifconfig -a
    
    snd2@snd2:~$ ifconfig -a
    enp3s0    Link encap:Ethernet  HWaddr 38:d5:47:8f:e7:0d  
              inet addr:192.168.0.2  Bcast:192.168.0.255  Mask:255.255.255.0


Then, change the interfaces file to create static IP address:

    sudo nano /etc/network/interfaces
    
        # interfaces(5) file used by ifup(8) and ifdown(8)
        # auto lo
        # iface lo inet loopback
        auto enp3s0
        iface enp3s0 inet static
        address 192.168.0.2
        netmask 255.255.255.0
        gateway 192.168.0.1


Finally, from another computer:

    ssh snd2@192.168.0.9 -p4321

and for filezilla:

    sftp://192.168.0.9







### RStudio server

From the RStudio website: https://www.rstudio.com/products/rstudio/download-server/

These commands were run to get rstudio-server installed and running:

     sudo apt-get install gdebi-core
     wget https://download2.rstudio.org/rstudio-server-1.1.442-amd64.deb
     sudo gdebi rstudio-server-1.1.442-amd64.deb

Then I went to

    http://<server-ip>:8787

in the browser and it just works.






























