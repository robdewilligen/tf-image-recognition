# Tensorflow Image Recognition script

# Results:
Iteration 1: without manipulation:  
Goal: Create a model capable of prediction on a split dataset
- accuracy: 0.9635, validation_accuracy: 0.6376  

Iteration 2: with data augmentation (flip, rotate and zoom)  
Goal: Improve the accuracy of the model by introducing more data by augmenting the available images
- accuracy: 0.7683, validation_accuracy: 0.7439

Iteration 3: with dropout regularization  
Goal: Improve the accuracy of the model by applying dropout regulation  
Note: due to the dropouts we need to increase the amount of epochs to get a better accuracy
- accuracy: 0.7862, validation_accuracy: 0.7289

The addition of the second and third iteration allow the models accuracy to be closer to the 
validation accuracy, which will give better results once the model will be used against unknown data
