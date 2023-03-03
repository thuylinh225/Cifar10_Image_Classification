# Cifar10_Image_Classification. 

## 1.1 Problem. 
Deep learning is a type of machine learning based on artificial neural networks that teaches computers to do what comes naturally to humans: learn by example. In deep learning, a computer model learns to perform classification tasks directly from images, text, or sound. Image classification is one of the most important applications of deep learning, refers to assigning labels to images based on certain characteristics or features present in them. The algorithm identifies these features and uses them to differentiate different images and assign labels to them. In this project, my goal is to classify images from CIFAR-10 dataset by training Convolutional Neural Networks (CNNs). To achieve this goal, I will:    
+ Load in the data.       
+ EDA - Inspect, Visualize, and Clean the data.      
+ Building and training models:     
    + Model 1: CNN with two layers 
    + Model 2: CNN with two layers adding dropout    
    + Model 3: CNN with three layers adding dropout 
    + Model 4: CNN with three layers adding Earlystopping and ReduceLROnPlateau 
    + Model 5: Resnet50 Model with "Imagenet" pretrained weights 
    + Model 6: Resnet101 Model with "Imagenet" pretrained weights 
+ Results and Analysis, compare 6 models. 
+ Build the best model with tuning hyperparameters. 
+ Conclusion.    

Here, I will compare these six deep learning models by roc_auc_score because it is generally seen as a more important measure of how good an algorithm is. This metric considers the trade-offs between precision and recall, while Accuracy only looks at how many predictions are correct. It tells us what is the probability that a randomly chosen positive instance is ranked higher than a randomly chosen negative instance. Since we care about true negatives as much as we care about true positives then it totally makes sense to use roc_auc_score. 

Reference Sources:    
(1) https://www.tensorflow.org/api_docs/python/tf/keras/datasets/cifar10/load_data    
(2) https://neptune.ai/blog/f1-score-accuracy-roc-auc-pr-auc 

## 1.2 Data 
In this project, I use CIFAR-10 dataset from keras datasets module.    

CIFAR-10 is a collection of images used to train Machine Learning and Computer Vision algorithms, used for object recognition. It is a subset of the 80 million tiny images dataset and consists of 60,000 32x32 color images containing one of 10 object classes, with 6000 images per class. It was collected by Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton.  

There are 50,000 training images and 10,000 test images, labeled over 10 categories in the official data.  

The label classes in the dataset are: 

![image](https://user-images.githubusercontent.com/63614659/221905625-09e1d667-cc0d-48ca-9e24-9655b64c758b.png)
