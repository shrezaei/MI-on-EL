# MI-on-EL
The implementation of "Accuracy-Privacy Trade-off in Deep Ensemble Learning"

## Library Versions
* Python 3.5
* Torch 1.6.0

## How to use this repository:
1. After downloading this repository, you need to also download a repository from Bearpaw that implemented model architectures and training:
```
$ git clone --recursive https://github.com/bearpaw/pytorch-classification.git bearpaw
```
2. You need to use bearpaw package to train models. Note that we assume an standard normalization of values between ([-1,1]). Make sure your model is trained with standart normalization or change the code. As an example, we included 10 resnet20 models trained on CIFAR10 in models/resnet20 path.
3. Now, you should run save_outputs.py to save the output of all models, before launching MI attack, as follows:
```
$ python save_outputs.py 
```
By default, it reads all models in the models/resnet20 path and stores the confidence values in the outputs/resnet20 path. You need use --models-path, to change the path to a directory where you stored your models. Set dataset and model architecture and other arguments, accordingly.
4. Now, run attack.py to run MI attack and report the MI attack on single model upto an ensemlbe of all models. 
```
$ python attack.py --output-type "confidence" --attack-type "aggregated"
```
