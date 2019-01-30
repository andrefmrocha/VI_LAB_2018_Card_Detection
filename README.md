# VI_LAB_2018_Card_Detection
Through the use of Siamese Networks for One Shot we have divised a model for image recognition used as a face recognition algorithm. This repository is said implementation for the NJU-ID Dataset, with card images and their respective faces.

## Colaborators
* [Andre Rocha](https://github.com/andrefmrocha)
* [Joana Ferreira](https://github.com/joanaferreira0011)
* [Manuel Coutinho](https://github.com/ManelCoutinho)
* [Lia Meireles](https://github.com/LiaMeireles)

## Requirements
* Tensorflow
* Keras
* [NJU-ID(Nanjing University ID Card Face Dataset)](https://cs.nju.edu.cn/rl/Data.html)

## One-Shot Learning
Despite most deep learning models needing a significant amounf of labeled samples per class, data acquisition is one of the hardest tasks, financially speaking. In order to combat the lack of data, learning from a few samples is much a more interesting approach than the acquisition of thousands of labeled samples. This is where one-shot learning shines further, the ability to learn from very few data is actually a great approach for many machine learning problems.

What is exactly a one-shot task? Given a test sample, an one-shot task would aim to classify this test image into one of several categories. For this support set of samples with a representing number of unique categories is given to the model in order to decide what is the class of the test images, without those samples having been previously seen by the model.

## Methodology

To solve this issue, it was selected the use of a Deep Convolutional Siamese Networks. Siamese Networks are two twin networks that accept distinct input but are joined in by a energy function that calculates a distance metric between the outputs of two nets. We have used a concatenation layer to join both networks calculating the euclidean distance between twin vectors.

<div style="text-align:center">
    <img src="https://i.imgur.com/1AdWbpg.png"/>
</div>

However, the use of Siamese Networks was still not achieving good enough results. The cause was attributed to the fact that the resolution of the image dataset was very low (160x160 px), so it was devised it would be best to use a autoencoder through the images beforehand.

An autoencoder is a type of neural network that encodes a set of data in order to reduce the dimension of the problem, being trained to ignore "noise". Alongside with the reduction, comes a reconstruction side, where the autoencoder tries to generate from the reduced encoding a representation as close as possible to its original input.

<div style="text-align:center">
    <img src="https://i.imgur.com/C49UqAx.png"/>
</div>

However autoencoders and decoders were used differently in this project, to try to cancel out background "noise" from the issue, after the autoencoders were trained with both the camera images and the card images, images would be encoded with their respective encoders; however, being decoded using the opposite decoder, meaning that card images were decoded using the camera decoder and, naturally, the camera images were decoded using the card decoder.

This attempt was indeed able to significantly improve our results.

## Code Details
There are two main files to run in this repo:
* CV_main.py which runs the actual Siamese Neural Network throughout the images
* AutoEnc_main.py which runs the autoencoder throught the images.

Regarding the rest of the code:
* CV_AutoEnc.py is the implementation of the autoencoder itself
* CV_CNN implements the Deep Convolutional Siamese Networks
* CV_Dataset handles the preparation of the dataset for both networks to use.


## Credits
All the colaborators would very much like to give credits to the ones who made all of this possible: the investigators from the Networked Intelligent Systems in the Telecommunications and Multimedia Centre of [INESC TEC](https://www.inesctec.pt/en), with a specific regard to those who accompanied us further:
* [Diogo Pernes](https://www.inesctec.pt/en/people/diogo-pernes-cunha)
* [Pedro Ferreira](https://www.inesctec.pt/en/people/pedro-martins-ferreira)
* [Ricardo Cruz](https://www.inesctec.pt/en/people/ricardo-pereira-cruz)

It is safe to say that without their help, guiding us with their expertise on the matter, this project would have never been achieved. 

