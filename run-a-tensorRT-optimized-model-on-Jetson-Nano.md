**Using TensorRT Command-Line Wrapper (trtexec):**

If you write Jet Pack 4.4+ image on the SD card, the TensorRT samples are at /usr/src/tensorrt/samples. 
At first, make sure your CUDA_INSTALL_DIR and CUDNN_INSTALL_DIR are all set. Then, we build the samples as follows:
```
cd /usr/src/tensorrt/samples
sudo make TARGET=aarch64
```
Now we can convert a pre-trained NN model to a TensorRT engine and run the inference on the Jetson Nano using the *trtexec* command.
```
cd /usr/src/tensorrt/bin
```
Let's download the architecture of a pre-trained model of ResNet50 from [https://drive.google.com/file/d/1X-RmNVEMYlak_HKWzUjCLgxpklHHjz-L/view?usp=sharing].
We can provide pre-trained weights as well via an argument to the trtexec command.

```
sudo trtexec --deploy=absolute/path/to/resnet50.prototxt --model=absoulte/path/to/resnet50.caffemodel --output=prob --batch=1 --saveEngine=resnet50.trt
```
This command will convert a ResNet50 pretrained model (including weights) to a corresponding TensorRT optimized model and save the engine as resnet50.trt.
Which we can run by the following command where the inference is fed with an arbitrarily generated input.
```
trtexec --loadEngine=resnet50.trt --batch=1
```
We can play with *trtexec* command arguments by studying ```trtexec --help``` command!
