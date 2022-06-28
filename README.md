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


Here, is what each notebook contains and the PyTorch constructs you will find in each of them:

## Basics [`↩`]
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

## Deep Learning for Vision Basics [`↩`]
* D2L_Conv_Basics.ipynb: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rajlm10/D2L-Torch/blob/main/D2L_Conv_Basics.ipynb)
  - Creating convolutions from scratch, padding, pooling and LeNet
  - Learn about `nn.Conv.2d`, `nn.AvgPool2d`
  
 * D2L_CNNs.ipynb: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rajlm10/D2L-Torch/blob/main/D2L_CNNs.ipynb)
    - Creating AlexNet, VGG-11, NiN (Network in Network), Google LeNet (Inception),Batch Normalization layer, ResNet and DenseNet from scratch
    - Learn about `nn.BatchNorm2d`, `nn.AdaptiveAvgPool2d` (Global Average Pooling)


## Deep Learning for Text Basics [`↩`]
 * D2L_Text_Basics.ipynb: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rajlm10/D2L-Torch/blob/main/D2L_Text_Basics.ipynb)
    - Creating a tokenizer and vocabulary, random & sequential sampling and Sequence Data Loader
    - Learn about corpus statistics (unigrams,bigrams,trigrams)

* D2L_Seq_Models.ipynb: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rajlm10/D2L-Torch/blob/main/D2L_Seq_Models.ipynb) 
    - character level RNNs,GRUs,LSTMs,stacked models, bidirectional models
    - Learn about `torch.nn.functional.one_hot`,`nn.RNN`,`nn.GRU`,`nn.LSTM` and when **NOT** to use bidirectional networks

* D2L_NMT.ipynb: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rajlm10/D2L-Torch/blob/main/D2L_NMT.ipynb) 
    - GRU and LSTM based encoder-decoder machine translation
    - Learn about generating dataloaders for machine translation, padding , truncating, end-to-end neural machine translation and BLEU evaluation

* D2L_NWKR.ipynb: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rajlm10/D2L-Torch/blob/main/D2L_NWKR.ipynb) 
    - Nadaraya Watson Kernel Regression, (non-parametric and parametric attention pooling)
    - Learn about an early form of attention for regression problems, both parametric and non-parametric.


* D2L_Attention_Mechanisms.ipynb: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rajlm10/D2L-Torch/blob/main/D2L_Attention_Mechanisms.ipynb) 
    - Different Attention Mechanisms used in Seq2Seq Models
    - Learn about Attention used in NMT, Bahdnau Attention (Additive Attention form) in GRU based Seq2Seq Models, Self Attention , Multi-Headed Attention, Positional Encodings (Traditional sinusoidal) and the Transformer.


## Optimization and Distributed Training [`↩`]
* D2L_Optimization.ipynb: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rajlm10/D2L-Torch/blob/main/D2L_Optimization.ipynb) 
    - Different Optimization scenarios and algorithms. 
    - Learn about GD, SGD, Variable Learning rates, Mini-batch GD ,Preconditioning , Momentum , AdaGrad , RMSProp, Adadelta, Adam , Yogi , Schedulers and Policies
    - Note that the book goes into much more mathematical detail and includes proofs (That I have left out).

* D2L_Distributed_Training.ipynb: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rajlm10/D2L-Torch/blob/main/D2L_Distributed_Training.ipynb) 
    - Speeding up training using distibuted GPU training, synchronizing computation, PyTorch's auto parallelism 
    - Learn about Hybrid Programming, Asynchronous Computation, Automatic Parallelism, Multi-GPU training, `nn.parallel.scatter`,`nn.DataParallel`,`torch.cuda.synchronize`
    - Note that a few sections in the notebook won't run on colab due to the requirement of multiple GPUs (atleast 2)

## Deep Learning Applications for Computer Vision [`↩`]
* D2L_Tuning.ipynb: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rajlm10/D2L-Torch/blob/main/D2L_Tuning.ipynb) 
    - Augment Datasets using  for robust training and learn how to fine-tune for image-classification.
    - Learn augmenting through `torchvision.transforms.RandomHorizontalFlip`,`torchvision.transforms.RandomVerticalFlip`,`torchvision.transforms.RandomResizedCrop`,
 `torchvision.transforms.ColorJitter` and setting multiple learning rates for training different parameters of the same network and a small trick to further improve fine-tuning image classes found in ImageNet.
 
 * D2L_Object_Detection.ipynb: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rajlm10/D2L-Torch/blob/main/D2L_Object_Detection.ipynb) 
    - Learn about multiscale object detection using a Single Shot Detector (from scratch). Note this is one of the tougher and more mathematical notebooks.
    - Learn about generating anchor boxes at different scales and ratios, sampling pixels uniformly from images as centres to generate anchor boxes, an algorithm to map the generated anchor boxes to ground truth bounding boxes,Non-maximal supression, the SSD architecture from scratch, and the object detection loss function.
    - This is one of the best chapters in the book since its very detailed and delves into the nitty-gritties of object detection. I spent a lot of time on this notebook due to its size. One tip I'd like to give here is keep track of the dimensions of tensors at all times since there are so many functions and transformations. Some functions in here are gold mines, they are scalable and blazing fast.


* D2L_Semantic_Segmentation.ipynb: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rajlm10/D2L-Torch/blob/main/D2L_Semantic_Segmentation.ipynb) 
    - Semantic Segmentation using a CNN on the PASCAL VOC 2012 dataset.
    - Learn about preprocessing data for semantic segmentation and transposed convolutions.
    - May require high RAM (unless you drastically reduce batch_size)

* D2L_Neural_Style_Transfer.ipynb: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rajlm10/D2L-Torch/blob/main/D2L_Neural_Style_Transfer.ipynb) 
    - Neural Style Transfer using a VGG-19.
    - Learn about the features needed to retain the content style and those to inject styles from the style image. Learn about the content loss and the gram matrix.

* D2L_CIFAR10_Challenge.ipynb: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rajlm10/D2L-Torch/blob/main/D2L_CIFAR10_Challenge.ipynb) 
    - Efficient Net B0 on CIFAR10. 
    - **Focus isn't on accuracy** but organizing the project,splitting the dataset properly, calculating mean and standard deviation for a dataset (to normalize the data) and using pretrained models.

* D2L_DogBreeds_Kaggle.ipynb: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rajlm10/D2L-Torch/blob/main/D2L_DogBreeds_Kaggle.ipynb) 
    - Deit tiny(FAIR) on Dog Breed Identification Dataset. 
    - Learn how to use HuggingFace and DEIT (a better performing variant of the vision transformer) on a custom dataset. Learn how to convert a custom image dataset to a HuggingFace image dataset and use the HuggingFace Trainer to train the model.
    - I have added the Deit model, the book uses a traditional pretrained CNN.


## Deep Learning Applications for NLP [`↩`]

* D2L_Word2Vec.ipynb: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rajlm10/D2L-Torch/blob/main/D2L_Word2Vec.ipynb) 
    - Train a Skip-gram model from scratch. 
    - Learn about sampling techniques, negative sampling for efficient training, word vectors, dealing with high-frequency words.

* D2L_Word_Vectors.ipynb: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rajlm10/D2L-Torch/blob/main/D2L_Word_Vectors.ipynb) 
    - Use GloVe and fasttext. 
    - Learn about using word embeddings to calculate similarities and analogies between words.
 
* D2L_Pretraining_BERT.ipynb: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rajlm10/D2L-Torch/blob/main/D2L_Pretraining_BERT.ipynb) 
    - Write BERT from scratch and pretrain it on Wiki-Text-2 using the MLM and NSP objective. 
    - Learn how to write BERT from scratch, how to use `torch.einsum` and coding the Masked Language Modelling objective and the Next Sentence Prediction Objective
    - One of my favourite chapters in the book!
    - I refactored the code in the book to incorporate `Einops` which makes it more readable, easier to understand, faster to code tensor operations and rearrangements and reduces the need to write a few extra classes and functions. I also decided to refactor the `Masked Softmax` class in the book and make it a part of the `MultiHeadAttention` Class
    - Note that since I had access to a Tesla V100, I decided to use a full scale BERT architecture but for the free colab tier, stick to a BERT with 2 layers, 128 hidden dims and 2 attention heads.

* D2L_Sentiment_Analysis.ipynb: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rajlm10/D2L-Torch/blob/main/D2L_Sentiment_Analysis.ipynb) 
    - Train a Bidirectional LSTM and a 1-D CNN (TextCNN) on the IMDB dataset and tune hyperparameters with Optuna. 
    - Learn how to use pretrained embeddings in your models, using a BiLSTM, multiple variations of 1-D CNNs including those with positional embeddings (sinusoidal and learnable). Also see how we can use Optuna to tune the hyperparameters of a model.
    - I have added the code to tune the hyperparameters (it is not a part of the book) and added a few variations based on the Exercise in the book 
    - Note that tuning hyperparameters requires a powerful GPU so feel free to skip that section.
 
* D2L_SNLI.ipynb: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rajlm10/D2L-Torch/blob/main/D2L_SNLI.ipynb) 
    - Train a decomposable attention model on the Stanford Natural Language Inference dataset. 
    - Learn how to use a MLP and attention to train an efficient model.

* D2L_SNLI_HuggingFace.ipynb: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rajlm10/D2L-Torch/blob/main/D2L_SNLI_HuggingFace.ipynb) 
    - Tuned [BORT]([url](https://github.com/alexa/bort)) from the Hugging Face hub on the Stanford Natural Language Inference dataset. 
    - Learn how to use a HuggingFace model, create a custom dataset compatabile with HuggingFace and use dynamic padding
    - You can use any model of your choice, I just wanted to experiment with BORT
    - I was very lucky to gain access to a NVIDIA A-100 40 GB GPU and would recommend that you significantly reduce the batch size in the notebook on regular colab instances.
    - Also note how always using a transformer model does not help. In the previous notebook, decomposable attention reached a much better accuracy score and more importantly took 45 minutes lesser to train.
 
## Generative Deep Learning [`↩`]
* D2L_GANs.ipynb: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rajlm10/D2L-Torch/blob/main/D2L_GANs.ipynb) 
    - An introduction to Generative Computer Vision, trained a GAN to sample out of a particular Gaussian, and a DCGAN to generate Pokemon 
    - Learn about Generators, Discriminators, an efficient way to write the training loop for GANs
    - Modified some of the code in the book to adopt some techniques from Sharon Zhou's CS236G (Stanford)

## Recommender Systems [`↩`]
- Work In Progress
- Porting MXNet code to vanilla PyTorch

* D2L_Matrix_Factorization.ipynb: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rajlm10/D2L-Torch/blob/main/D2L_Matrix_Factorization.ipynb) 
    - Matrix Factorization on MovieLens100k
    - Learn about the Matrix Factorization algorithm, creating the dataset and dataloaders for Recommender System
