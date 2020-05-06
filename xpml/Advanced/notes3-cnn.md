---
title: "Convolutional Neural Networks"
geometry: margin=2cm
output: pdf_document
---

# Contents

* Convolutional Neural Networks
    * Examples
        * LeNet
        * AlexNet
    * Dense vs. Convolutional
        * Dense Layers on Images
        * Convolutional Layers on Images
        * Activation Maps
        * Visualizing Weights and Maps
    * Architecture 
        * Convolutional
        * Pooling
        * Dense Relu
    * Architectures
        * AlexNet in Detail
        * ResNet in Detail
        * VGG, GoogLeNet, et al.
    * Pre-Trained Networks
        * Feature Engineering
        * Transfer Learning



$\pagebreak$ 


# Overview

Aim: to explicitly introduce techniques to model structure within the input. 

http://scs.ryerson.ca/~aharley/vis/conv/flat.html

https://cs.stanford.edu/people/karpathy/convnetjs/


$\pagebreak$ 


# CNNs for Images

* Image Classification (global labelling)
* Object Detection (bounding box)
* Segmentation (outlining) 
* Pose-Estimation
* Video 
* Game Play (CNN with Reinforcement Learning)
* Image Captioning (CNN with RNN)
* Style Transfer


$\pagebreak$ 


# History

Adaline, c. 1960s -- recognisable network structure but no backprog (and therefore no workable training strategy).  Rumelhart, c. 1980s -- now with backprop. 

Hinton, c. 2006 -- deep network structure shown to be trainable (very brittle weight initialization strategy). 

LeNet, c. 1998 -- first performant CNN network for image classification (zip code recognition). 

AlexNet, c. 2012 -- CNN for image classification shows deep learning most performant algorithm; significant jump.  


$\pagebreak$ 


# Dense Layers on Images

Images are aligned along a 1D vector, and dot-producted with the weights, so that the weights are connected to pixels at widely geometrically distinct locations. (Incidentally, this means that permutations of the input are as trainable as unpermuted input -- so that a picture of a '1' and a picture of all the pixels of '1' scattered are not distinct). 


![Random MNIST with Dense Network](images/rnd-mnist.png)

![Random MNIST with Dense Network -- Loss](images/rnd-mnist-train.png)


$\pagebreak$ 


# Convolutional Layers on Images

Convolutional layers take a whole-image input in 2D matrix form (for a single black-and-white image), and 3D-tensor form for red/green/blue layers. The input layer, and convolutional layers, therefore have a **width**, **height** *and* **depth**.


![Dense Network vs. CNN](images/cnndepth-vs-nn.png)

Now the weights are arranged in a filter (a matrix) which is dot-producted with regions of the image, ie., the filter size. The filter is moved across the entire image, yielding a new layer (activation map) which tracks the local, regional response of the image to that filter. 


![LeNet Architecture](images/lenet-diag.png)


$\pagebreak$ 



# Filters and Activation Maps

Each filter (trainable weight matrix) is a kind of template which tracks whether a region in the input matches it (ie., the activation is maximal where that weight matrix encounters its target pattern).

![Convolution with Linear Input](images/convolution-linear.png)

The goal of CNN learning is to produce filters which map the input space to a new feature space. The hope is that this new feature space is linearly seperable. Its typical to have $2^n$ filters, eg., $32, 64, ... 512 ...$.

![Prior Layers in a CNN composing to shapes in later layers](images/distill-feature-vis.png)

An *activation map* records the response of the input at every region to a single filter. With, eg., 10 filters, there are 10 activation maps.

The activation maps are together the *new representation* of the input image. Given $32$ filters, there will be $32$ activation maps. 


$\pagebreak$ 


# Filter Heirachies 

In deep CNNs, early filters tend to learn simple shapes and latter filters learn combinations of those simpler shapes. 

![Early Weights](images/weights-early.png)

![Late Weights](images/weights-late.png)




$\pagebreak$ 


# Micro-Architecture 

Conv layers are usually sequences with ReLU dense layers -- where the dense layer effectively takes the Conv layer as a new feature space to fit over.

Pooling layers reduce the size of this feature space (*down-samples*), both to regularize and to limit the number of parameters in the system to reduce training time.


![VGG Model Summary](images/vgg_model_summary.png)




$\pagebreak$ 


# Convolutional Layers


![Inputs with filters applied](images/conv-example.png)


* 7x7x1 input ($X_W = 7, X_D = 1$)
* 5x5 filter ($F_W = 5$)
* 1 stride ($S = 1$)
* 1 filter ($N_F = 1$)

* 7x7x1 input
* 5x5 filter
* 2 stride

* 3 stride
    * does not fit, so not used
    * but: pad input with 0s, and then use 3

* output size: $\frac{X_W - F_W}{S+1}$ 

Padding is also used just so the output is the same size as the input. As convolving reduces output size, it effectively pools -- down-sampling. If you have a deep network, many unpadded convolutional layers would reduce the ability of the network to learn (ie., reduce the number of parameters).

The total number of parameters introduced by a convolution layer is:
$weights = N_F F_WF_H X_D$ + $biases = N_F$


$\pagebreak$ 


# NNs and CNNs in Python


```python
model = Sequential([
    Dense(512, input_shape=(784,), activation='relu'),
    Dense(10, activation='softmax'),
])
model.compile("adam", "categorical_crossentropy",
              metrics=['accuracy'])
```
```
_____________________________________________________
Layer (type)          Output Shape         Param #
=====================================================
dense_7 (Dense)       (None, 512)           401920
_____________________________________________________
dense_8 (Dense)       (None, 10)            5130
=====================================================
Total params: 407,050
Trainable params: 407,050
_____________________________________________________
```

```python
num_classes = 10
cnn = Sequential()
cnn.add(Conv2D(32, kernel_size=(3, 3),
               activation='relu',
               input_shape=input_shape))
cnn.add(MaxPooling2D(pool_size=(2, 2)))
cnn.add(Conv2D(32, (3, 3), activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2, 2)))
cnn.add(Flatten())
cnn.add(Dense(64, activation='relu'))
cnn.add(Dense(num_classes, activation='softmax'))
```
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d_1 (Conv2D)            (None, 26, 26, 32)        320
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 13, 13, 32)        0
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 11, 11, 32)        9248
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 5, 5, 32)          0
_________________________________________________________________
flatten_1 (Flatten)          (None, 800)               0
_________________________________________________________________
dense_9 (Dense)              (None, 64)                51264
_________________________________________________________________
dense_10 (Dense)             (None, 10)                650
=================================================================
Total params: 61,482
Trainable params: 61,482
_________________________________________________________________
```


$\pagebreak$ 


# Pooling Layers

Spatial down-sampling, reduces $X_W, X_H$ but not $X_D$.

The most common pooling layer is a MaxPooling which keeps the maximum value in some region (eg., in a 2x2 area).

![Max Pooling](images/maxpool.png)

2x2 is common, reduces output size by half. 



$\pagebreak$ 

# Architectures

![VGG vs. AlexNet -- NB. Simplicity of VGG16](images/network-comparison.png)


$\pagebreak$ 

# LeNet

![LeNet](images/lenet-diag.png)


$\pagebreak$ 

# AlexNet
![VGG16 vs. AlexNet](images/other_architectures.png)

![AlexNet Results (from paper)](images/alexnet-results.png)




$\pagebreak$ 

# ResNet 

![ResNet vs. VGG](images/vgg-vs-resnet-architecture.png)


```python
base_model = applications.resnet50.ResNet50(
    weights= None, include_top=False, input_shape=(img_height,img_width,3)
)
```


$\pagebreak$ 


# Residual Networks


**INSERT ResNet path diagram cf. https://www.coursera.org/learn/convolutional-neural-networks/lecture/HAhz9/resnets**


**INSERT dense -> residual diagram**

We can't fit deep networks well - not even on training set!

"vanishing gradient problem" - was motivation for relu, but not solved yet.


The deeper the network gets, usually the performance getsbetter. But if you make your network too deep, then you
can't learn it anymore.


![Effect of Deepening on Loss](images/resnet-no-deep-nets.png)
![Effect of Residual Layers on Loss](images/resnet-success.png)


This is on CIFAR-10, which is a relatively small dataset.But if you try to learn a 56-layer convolutional, you cannot
even optimize it on the training set. So basically, it's notthat we can't generalize, we can't optimize. So these are
universal approximators, so ideally, we should be able tooverfit completely the training set.

But here, if we make it too deep, we cannot overfit thetraining set anymore. It's kind of a bad thing. Because wecan’t really optimize the problem. So this is sort ofconnected to this problem of vanishing gradient that it'svery hard to backpropagate the error through a very deep netbecause basically, the idea is that the further you get fromthe output, the gradients become less and less informative.We talked about RELU units, which sort of helped to makethis a little bit better. Without RELU units, you had like 4or 5 layers, with RELU units, you have like 20 layers. Butif you do 56 layers, it's not going to work anymore.
Even it's not going to work anymore, even on the trainingset. So this has been like a big problem. And it has asurprisingly simple solution, which is the RES-NET layer.

$\pagebreak$

# Residual Layers

A building block of a ResNet is called a residual block or identity block. A residual block is simply when the activation of a layer is fast-forwarded to a deeper layer in the neural network.

This simple tweak allows training much deeper neural networks.
In theory, the training error should monotonically decrease as more layers are added to a neural network. In practice however, for a traditional neural network, it will reach a point where the training error will start increasing.
ResNets do not suffer from this problem. The training error will keep decreasing as more layers are added to the network. In fact, ResNets have made it possible to train networks with more than 100 layers, even reaching 1000 layers.


```python
# residual layer:
X_pre = X
X_post = CNN(BN(X_pre)) # suppose learn (w,b) = 0
X = Add()([X, X_post]) # this skip connection then adds 0 to X
# moving on...
```


## How do skip connections work?

Extra layerse in dense networks make learning worse because they can choose weights/biases that misperform -- worst case for a residual layer is learning identity, so layer does not make existing predictions worse.

* Mitigate vanishing gradient by forwarding pre-layer activations (boosting the weights)
* Worst-case layer performance is "identity", ie., leaving activations unchanged

* therefore: depth without vanishing gradient / learning degredation







$\pagebreak$ 


# Deep Architectures in Python

## Simple Dense with Keras Functional API

```python

from keras.models import Model
# This returns a tensor
inputs = Input(shape=(784,))

# a layer instance is callable on a tensor, and returns a tensor
x = Dense(64, activation='relu')(inputs)
x = Dense(64, activation='relu')(x)

predictions = Dense(10, activation='softmax')(x)

# This creates a model that includes
# the Input layer and three Dense layers
model = Model(inputs=inputs, outputs=predictions)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(data, labels)  # starts training

```

$\pagebreak$

## CNN with Keras Functional API

```python

from keras.layers import Input, Conv2D, MaxPooling2D, Flatten
from keras.models import Model

num_classes = 10

inputs = Input(shape=(28, 28, 1))
conv1_1 = Conv2D(32, (3, 3), activation='relu',padding='same')(inputs)
conv1_2 = Conv2D(32, (3, 3), activation='relu',padding='same')(conv1_1)
maxpool1 = MaxPooling2D(pool_size=(2, 2))(conv1_2)
conv2_1 = Conv2D(32, (3, 3), activation='relu',padding='same')(maxpool1)
conv2_2 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv2_1)
maxpool2 = MaxPooling2D(pool_size=(2, 2))(conv2_2)
flat = Flatten()(maxpool2)
dense = Dense(64, activation='relu')(flat)
predictions = Dense(num_classes, activation='softmax')(dense)

model = Model(inputs=inputs, outputs=predictions)
```

$\pagebreak$

## CNN with Residual (skip) Connections 

```python


from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, add
from keras.models import Model

num_classes = 10
inputs = Input(shape=(28, 28, 1))

conv1_1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
conv1_2 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1_1)

maxpool1 = MaxPooling2D(pool_size=(2, 2))(conv1_2)

conv2_1 = Conv2D(32, (3, 3), activation='relu', padding='same')(maxpool1)
conv2_2 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv2_1)

skip2 = add([maxpool1, conv2_2])

maxpool2 = MaxPooling2D(pool_size=(2, 2))(skip2)
flat = Flatten()(maxpool2)
dense = Dense(64, activation='relu')(flat)
predictions = Dense(num_classes, activation='softmax')(dense)
model = Model(inputs=inputs, outputs=predictions)

```

```
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_4 (InputLayer)            (None, 28, 28, 1)    0                                            
__________________________________________________________________________________________________
conv2d_13 (Conv2D)              (None, 28, 28, 32)   320         input_4[0][0]                    
__________________________________________________________________________________________________
conv2d_14 (Conv2D)              (None, 28, 28, 32)   9248        conv2d_13[0][0]                  
__________________________________________________________________________________________________
max_pooling2d_5 (MaxPooling2D)  (None, 14, 14, 32)   0           conv2d_14[0][0]                  
__________________________________________________________________________________________________
conv2d_15 (Conv2D)              (None, 14, 14, 32)   9248        max_pooling2d_5[0][0]            
__________________________________________________________________________________________________
conv2d_16 (Conv2D)              (None, 14, 14, 32)   9248        conv2d_15[0][0]                  
__________________________________________________________________________________________________
add_2 (Add)                     (None, 14, 14, 32)   0           max_pooling2d_5[0][0]            
                                                                 conv2d_16[0][0]                  
__________________________________________________________________________________________________
max_pooling2d_6 (MaxPooling2D)  (None, 7, 7, 32)     0           add_2[0][0]                      
__________________________________________________________________________________________________
flatten_2 (Flatten)             (None, 1568)         0           max_pooling2d_6[0][0]            
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 64)           100416      flatten_2[0][0]                  
__________________________________________________________________________________________________
dense_2 (Dense)                 (None, 10)           650         dense_1[0][0]                    
==================================================================================================
Total params: 129,130
Trainable params: 129,130
Non-trainable params: 0
__________________________________________________________________________________________________
```



$\pagebreak$ 


# Pre-Trained Networks

![Frozen/Unfrozen Layers in a Deep NN](images/frozen-train.png)

Reusing networks trained on one style of input can restyle other images:

![Style Transfer](images/style-transfer.png)