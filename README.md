# D2L-Torch
Learning PyTorch through the [D2L](https://d2l.ai/) book. A series of notebooks for the same

```
@article{zhang2021dive,
    title={Dive into Deep Learning},
    author={Zhang, Aston and Lipton, Zachary C. and Li, Mu and Smola, Alexander J.},
    journal={arXiv preprint arXiv:2106.11342},
    year={2021}
}
```

Hi there!
I created this repo, as a code along space for the textbook. The book's original [Github](https://github.com/d2l-ai/d2l-en) can be found here. I have made a few modifications to the code in the book and added comments and references from the text in the notebooks wherever I thought would be helpful. I have also debugged some of the issues I ran into while running original code.

_**Since I have some experience in Deep Learning (with Tensorflow), my aim here was to learn PyTorch through the book and I'd recommend this repo to those who want to learn torch and already know deep learning in other frameworks. For absolute beginners I'd strongly recommend reading the book**_

**Note that I have only added notebooks that I felt could be used as boilerplate to build models, hence I have omitted some of the didatic code (eg. creating models and algorithms from scratch) and used high level torch APIs wherever available which I think makes the code perfect to be used as a boilerplate in terms of neatness and efficiency.**



Here, is what each notebook contains and the PyTorch constructs you will find in each of them:
* D2L_LNN.ipynb: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rajlm10/D2L-Torch/blob/main/D2L_LNN.ipynb)
  - A simple linear regression model 
  - Learn about `torch.utils.data.TensorDataset`,`torch.utils.data.DataLoader`,`nn.Sequential`,`nn.Linear`,`nn.MSELoss`,`torch.optim.SGD`
  
* D2L_SR.ipynb: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rajlm10/D2L-Torch/blob/main/D2L_SR.ipynb)
  - A simple softmax regression model on Fashion MNIST 
  - Learn about `torchvision.Transforms`,`torchvision.datasets`,`nn.CrossEntropyLoss`,`nn.init.normal_`

* D2L_MLP.ipynb: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rajlm10/D2L-Torch/blob/main/D2L_MLP.ipynb)
  - A simple Multilayer Perceptron model on Fashion MNIST 
  - Learn about `nn.Flatten`,`nn.ReLU`

* D2L_Dropout.ipynb: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rajlm10/D2L-Torch/blob/main/D2L_Dropout.ipynb) 
  - A Multilayer Perceptron model with Dropout on Fashion MNIST 
  - Learn about `nn.Dropout`,`nn.ReLU` and how to create a Dropout Layer from scratch

* D2L_looking_into_torch.ipynb: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rajlm10/D2L-Torch/blob/main/D2L_looking_into_torch.ipynb)
  - Some important utilities and creating custom layers
  - Learn about `nn.init.xavier_`, Extracting network parameters in different ways, initializing shared parameters, Xavier initialization, subclassing nn.Module,saving and loading tensors/whole models and GPUs
  - The runtime error here is **intentional** and serves to show why tensors must be on the same device (same CPU or same GPU) during computations.

* D2L_Conv_Basics.ipynb: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rajlm10/D2L-Torch/blob/main/D2L_Conv_Basics.ipynb)
  - Creating convolutions from scratch, padding, pooling and LeNet
  - Learn about `nn.Conv.2d`, `nn.AvgPool2d`
  
 * D2L_CNNs.ipynb: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rajlm10/D2L-Torch/blob/main/D2L_CNNs.ipynb)
    - Creating AlexNet, VGG-11, NiN (Network in Network), Google LeNet (Inception),Batch Normalization layer, ResNet and DenseNet from scratch
    - Learn about `nn.BatchNorm2d`, `nn.AdaptiveAvgPool2d` (Global Average Pooling)

 * D2L_Text_Basics.ipynb: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rajlm10/D2L-Torch/blob/main/D2L_Text_Basics.ipynb)
    - Creating a tokenizer and vocabulary, random & sequential sampling and Sequence Data Loader
    - Learn about corpus statistics (unigrams,bigrams,trigrams)

* D2L_Seq_Models.ipynb: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rajlm10/D2L-Torch/blob/main/D2L_Seq_Models.ipynb) 
    - character level RNNs,GRUs,LSTMs,stacked models, bidirectional models
    - Learn about `torch.nn.functional.one_hot`,`nn.RNN`,`nn.GRU`,`nn.LSTM` and when **NOT** to use bidirectional networks

  
