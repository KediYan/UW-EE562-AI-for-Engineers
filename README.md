# Matryoshka Representation Learning for Flexible Approximate Nearest Neighbors Search

## How to Run the Code
The main file is the Final Project.ipynb
### change file locations
To implement the code either locally or on Google Colab, **any code that contains the location of the files should be edited accordingly.** For example:
```python
#CHANGE  
%cd /content/gdrive/MyDrive/"Colab Notebooks"/"UW EE562 AI for Engineers"/HW6-FinalProject/MRL/
#TO
%cd /content/gdrive/MyDrive/"Colab Notebooks"/YOUR-LOCATION # if operate on Google Colab
#or
%cd /YOUR-LOCATION # if operating locally 
```
### change dataset
To use CIFAR-100, change the dataset parameter to _cifar_. Otherwise, change it to _imagenet_.
```python
''' WRITE TRAIN FFCV FILE '''
!python write_imagenet.py \
        --cfg.dataset=cifar \ # change this to imagenet for different ResNet50 models with Imagnet dataset
        ...
''' WRITE VAL FFCV FILE '''
!python write_imagenet.py \
        --cfg.dataset=cifar \ # change this to imagenet for different ResNet50 models with Imagnet dataset
        ...
```
### change training parameters
- modify the model that will be tested
```python
  --model.YOUR-PARAMETER
```
- change the parameter based on your hardware specs
```python
  --data.in_memory=1
  --dist.world_size=1
  --training.distributed=0
  --training.batch_size=16
  --validation.batch_size=16
```
- the Imagenet-1K dataset has an output classes of 1000. The CIFAR-100 dataset has an output classes of 100. This is a hard-code parameter in the _train_imagenet.py_
```python
model.fc = MRL_Linear_Layer(self.nesting_list, num_classes=100, efficient=self.efficient)
model.fc =  FixedFeatureLayer(self.fixed_feature, 100)
```
- the hyperparameters can be trained
```python
  --training.YOUR-PARAMETER
  --lr.YOUR-PARAMETER
```
