# NN-layers-implemenation

## 1 Intro
**MNIST** digits dataset is a widely used dataset for image classiﬁcation in machine learning ﬁeld. It contains 60000 training examples and 100000 testing examples. The digits have been size-normalized and centered in a ﬁxed-size image. Each example is a 784×1 matrix, which is transformed from an original 28×28 grayscale image. Digits in MNIST range from 0 to 9. Some examples are shown below.

<p align="center">
  <img width="300" height="300" src="https://cdn-images-1.medium.com/max/800/0*At0wJRULTXvyA3EK.png">
</p>

In this project, have been built a **Softmax Classiﬁer**, **Multilayer Perceptron(MLP)** and **ConvNet** to perform MNIST classiﬁcation respectively.

## 2 Softmax for MNIST Classiﬁcation

### 2.1 FilesDescription 
- **softmax.ipynb** is an IPython Notebook ﬁle which describes the main contents of this project. Data loading, hyerparameters setting, training and testing are included in this ﬁle.
- **mnist_data_loader.py** is used to load MNIST dataset.
- **softmax_classiﬁer.py** describes the softmax classiﬁer.

## 3 MLP for MNIST Classiﬁcation

### 3.1 FilesDescription 
- **MLP.ipynb** describes the main contents of this project.
- **network.py** describes network class, which can be utilized when deﬁning network architecture and performing model training.
- **optimizer.py** describes SGD optimizer class, which can be used to perform forward and backward propagation.
- **solver.py** describes training and testing pipeline.
- **plot.py** describes plot_loss_and_acc function which can be used to plot curves of loss and accuracy. 

In addition, there are several layers deﬁned in criterion/ and layers/. Our implementation is guided by modularity idea. Each layer class has three methods: __init__, forward and backward. __init__ method is used to deﬁne and initialize some parameters. forward and backward are used to perform forward and backward propagation respectively. 
- **FCLayer** treats each input as a simple column vector (need to reshape if necessary) and produces an output vector by doing matrix multiplication with weights and then adding biases: u = Wx + b.
- **SigmoidLayer** is a sigmoid activation unit, computing the output as f(u) = 1/(1+exp(−u)).
- **ReLULayeris** is a linear rectiﬁed unit, computing the output as f(u) = max(0,u). 
- **EuclideanLossLayer** computes the sum of squares of differences between inputs and label.
- **SoftmaxCrossEntropyLossLayer** can be viewed as a mapping from input to a probability distribution.

## 4 ConvNet for MNIST Classiﬁcation

Since ConvNet operates on two-dimensional input, we need to convert the raw data of 784×1 vector into 28×28 matrix. The data preprocessing and conversion have been done in the starter code.

### 4.1 FilesDescription 
- **homework_3.ipynb** describes the contents of this project.
- **conv_layer.py** implements the convolutional layer. It consists of trainable weight W and bias b. W is stored in 4 dimensional matrix with dimensions (n_out, n_in, kh, kw), where kh and kw speciﬁes the height and width of each ﬁlter (also called kernel size), n_in denotes the channels number of input which each filter will convolve with, and n_out denotes the number of the filters. There is another important parameter pad which speciﬁes the number of zeros to add to each side of the input. Therefore the expected height dimension of output should be equal to H + 2 × pad−kernel_size + 1 and width likewise.
- **pooling_layer.py** is the MaxPoolingLayer and has only been implemented in non-overlapping style (stride=kernel_size). Therefore the expected height dimension of output should be equal to (H + 2 × pad) / kernel_size and width likewise.
- **dropout_layer.py** implements Dropout layer.
- **reshape_layer.py** is used to reshape feature maps and δ.
