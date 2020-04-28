# CovidAI
Automatic Covid19 detection using patient lung X-rays, implemented with ResNet152 transfer learning on a PyTorch framework. <br />
Team members: Ricky Ma, Kaiwen Hu

## Introduction
The COVID-19 X-ray classification problem is a binary classification problem using X-ray images of patients' chests. Using these images, our goal is to train a machine learning model to determine whether a patient has COVID-19 or not. As the situation is still developing, the available public datasets of Covid-19-positive lung X-rays are limited. The provided dataset contains only 70 training images and 20 test images. With such a small dataset, traditional DNN or CNN approaches will not work. Hence, we approach this classification problem with transfer learning using various pretrained image classification models provided by Torchvision.

## Method
With a relatively small dataset, we decided to use Transfer Learning rather than training a CNN from scratch. Transfer Learning uses preexisting CNN that has already been trained on relatively large amount of images. Torchvision, a package provided by PyTorch, contains various datasets, model architectures, and image transformations for computer vision and image classification. We used \textbf{various pretrained CNNs for fixed feature extractions} and then \textbf{fine-tuned the CNNs}.

To use a CNN as fixed feature extractor, we take a CNN pretrained on a large dataset and remove the last fully connected layer. For example, applying transfer learning on ImageNet (a CNN containing 1.2 million images with 1000 classes), only the last fully connected layer is randomly initialized. This layer is the only one that is trained, and the remaining layers are kept as fixed parameters.

Fine-tuning a CNN not only extracts the very last fully connected layer, but also fine-tunes the weights of the fixed layers via backpropagation. The motivation behind this is to set the parameters of the CNN to be more specific towards the task at hand. For example, a major proportion ImageNet's data are images of dogs and their respective breeds, so the ImageNet's weights could be more biased towards classifying between different dog breeds. To mitigate this effect, fine-tuning the CNN helps with training the model for our specific classification problem.

Hence, our approach to Covid-19 diagnosis via lung X-rays starts with a pretrained image classification CNN. The last fully connected layer is replaced with a new layer with randomized weights. The CNN is then fine-tuned by training it on our lung X-ray data using cross entropy loss and SGD with a step learning rate. The CNN is trained for a set number of epochs and the resulting model is used to predict the labels of the 20 test images.

## Experiments
A number of pretrained CNNs provided by torchvision were tested and compared. Out of AlexNet, VGG, DenseNet, and ResNet architectures, ResNet was the only model able to generate predictions that contained labels for both categories. All other CNN architectures predicted that all test images were Covid-19 negative. Out of ResNet architectures, ResNet152 achieved the highest score. 

To improve training performance, 20\% of the training data were used as a validation dataset, resulting in 56 training images and 14 validation images for each epoch. A validation split of 10\% resulted in little improvement to test accuracy, while a split of more than 20\% decreased training efficiency. The max number of epochs was set at 30, as validation loss plateaued around epoch 25. 

Stochastic gradient descent was picked for its relatively fast optimization speed. We included Nesterov momentum to accelerate the convergence on the minima. Step decay was also added to our SGD algorithm to improve model accuracy. Step sizes of 7 worked well, and were scheduled via PyTorch's learning rate scheduler. 

## Results
| Model                             | Kaggle Score  |
| -------------                     |:-------------:|
| transfer learning using ResNet152 | 0.82352       |

## Conclusion
There were number of teams that outperformed our result. This is probably because our model implemented Transfer Learning using naive approaches for fixed feature extractor and fine-tuning. Thus, our CNN may have been a suboptimal model for classifying whether a patient had COVID-19 or not through X-ray image recognition. We could have approached to solve the problem effectively if we had considered more carefully which weights of the layers to manipulate. 

Additionally, more time could have been spent on hyperparameter tuning instead of comparing the various architectures. In hindsight, we should have stuck with a ResNet model and focus more on adjusting learning rates, momentum, batch-sizes, epochs, and other hyperparameters. Different gradient descent optimization algorithms could have been tested, like Adadelta, RMSprop, or Adam. Nevertheless, in this specific setting, we thought as machine learning consultants. False positive results would not be penalized as much as false negative results.

Through this binary image classification problem, we learned that CNNs used in the real world are seldom trained from scratch. It is not very practical for a variety of reasons - with the leading cause being datasets that are too small. Additionally, classification via transfer learning achieves sufficient, or even ideal, results with minimal time and effort. Thus, many open source packages are available for deep learning engineers to solve problems by transforming pre-existing CNN models to do the task at hand.
