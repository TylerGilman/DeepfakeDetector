# Deepfake Detection
## Overview
Misinformation on the internet has increasingly become a topic of concern in recent years, and deepfakes are a very effective and easy way to spread misinformation online. With the sheer volume of videos online and rapidly evolving deepfake technology, it is impossible to mannually filter deepfakes on the internet. With this project, we hope to develop a solution to this problem via a system of neural networks that when feed videos or images can categorize them as real or fake. We have developed a convolutional neural network for categorizing potentially tampered images and a recurrent neural network for categorizing potentially tamplered videos. All models were frained using the FaceForensics++ dataset. 

## Results and Model Performancs
### Convolutional Neural Netwrk (Images)
After training for 35 epochs, our best performing  model achieved 97.17% accuracy on real images and 98.89% accuracy on deepfaked images. Despite this, we foundthat the model does not perform as well on data outside of the FaceForensics++ dataset.

### Recurrent Neural Network (Videos)
After training for 100 epochs on 288 real and 288 deepfaked videos, the best performing RNN achieved a less than stellar 46% accuracy.

## Limitations
One of the major limitations with this project is the dataset. Deepfake video datasets are few and hard to come by due in part to copyright restrictions on videos. On top of that video data takes up a lot of disk space, and it is slow to work with, which was a limiting factor in our training.

We believe that, due to the model's inconsistant performance on images from outside of the FaceForensics++ dataset, our best performing convolutional neural network is overfitted to the data. A natural way to try and combat this would be to expand the training data, but as listed above, training data is hard to get. 

For the recurrent neural network, the poor performace could be due to a vaiety of factors. The recurrent model is dependant on the performance of the convolutional network, so if the convolutional network is indeed overfitted, that could impact performance here. The hyperparameters of the model may also need to be adjusted more thoroughly. As previously stated, working with video data takes a long time, so the amount of tuning that could be done was limited. It could also simply be the case that a recurrent neural network is not the best apporach to solving this problem. 

## Running the Network
To feed a potentially deepfaked image or video to the convolutional or recurrent neural network respectively, use the NetworkTest.py file. Run the file and input the file location of either a video or image to get the model's classification (it may take a moment to prompt you to enter a path). Input "stop" to stop the program. 
