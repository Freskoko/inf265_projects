# INF265: Project 1

## 1.

### Backpropogation

### Gradient descent

## 2.

### Machine learning pipelines

## 3.

We split the data up, 0.90% train, and validation / test being 0.05
We did this because neural networks are data hungry. Validation and test may have a low amount of data, but we still beleive they can help us find a good model.

We preprocessed by normalizing (based on train only).

We did some data exploration.
First off we checked the class distribution:

[class_dis](imgs/class_dist.png)

Accuracy will be good here.

Looking at some of the (normalized) images:


[plane](imgs/plane_train.png)
[bird](imgs/bird_train.png)



### Results

We had 4 different models, a baseline, a deep one, a wide one, and a model with dropout enabled.

Hyperparams chosen: 

```python
learning_rate_ls = [1e-3, 1e-2]
momentum_ls = [0.5, 0.9]
weight_decay_ls = [1e-4, 1e-2]

# only for dropout model
dropout_rate_ls = [0.2, 0.4]  # wanted to try 0.8 to see the effect

MAX_EPOCHS = 30

# we check model performance at 5,15, and 30 epochs
EPOCH_INVESTIGATE_POINTS = [5, 15, 30]
```

We chose these in order to have a wide amount of hyperparameters, without having too many, causing training to take a long time.

Lets see how the different models performed.

### Baseline:

*Architecture*:
```text
inp layer: 3072 - 512 -> relu
hid layer: 512 - 128 -> relu
hid layer: 128 - 32 -> relu
out layer: 32 - 2 no activation function
```

[baseline](imgs/hyperparams_MyMLPBaseLine.png)

We see a lot of different behviour in this model, depending on the hyperparameters.
An interesting model is the one in the bottom left, this model has a high learning rate at 0.01 and a lot of momentum. What we see is that the model overfits as its training loss goes down a lot. This makes sense with its hyperparameters, as when the model starts to move in one direction, it can be hard to change its direction. Interestingly, as the model's validation loss increases, but the validation accuracy does not change much over the epochs.
This is a good example of how a loss function and performance metric are not the same thing!

The loss increasing means that the model is less and less sure of the correct target, but it is atleast over 50% since the validation accuracy is decent.

THe model in the middle bottom, (8th image) seems to have issues learning. --
<!-- todo add one mroe -->


Generally, these models all learn, but some learn slower


### Deeper model:

*Architecture*:
```
inp layer: 3072 - 512 relu
hid layer: 512 - 128 relu
hid layer: 128 - 64 relu
hid layer: 64 - 32 relu
out layer: 32 - 2 no activation function
```

This deeper network has an additional layer. This can help the model learn more complex features, but may also increase overfitting (a regular risk when making a model more complex). 

[deep](imgs/hyperparams_MyMLPDDeep.png)

Some of the models, with the bottom left model in particiular, are overfitting. This model (,uch like the top left in the baseline) sees a steep decline in training loss (and a correspoding HIGH training accuracy), but the validation loss is gradually increasing. The validation accuracy is actually quite good, but as disucssed, these are not the same.
The deeper model can capture more of the image features, and more of its weights can adjust to the training data, learning it better. It's weights with a high learning rate (0.01) and momentum (0.9) support this aggressive learning style.

<!-- todo -->
Compare top left and top middle, similiar but weight decay.

### Wider model:

*Architecture*:
```text
inp layer: 3072 - 512 -> relu
hid layer: 512 - 256 -> relu
hid layer: 256 - 32 -> relu
out layer: 32 - 2 no activation function
```

This wider network has a layer with an increased size. This can help the model learn more complex features, but may (much like the deeper network) also increase overfitting (a regular risk when making a model more complex). 

[wide](imgs/hyperparams_MyMLPWide.png)


### Model with dropout:

*Architecture*:
```
inp layer: 3072 - 512 relu (dropout)
hid layer: 512 - 128 relu (dropout)
hid layer: 128 - 32 relu (dropout)
out layer: 32 - 2 no activation function (no dropout)
```

This network is the same as the basline, but has dropout between each layer (except first -> second last -> output). Dropout is a form of regularization. What can happen in a network without dropout, is that two or more neurons can become very codependent. A neuron may focus a lot on a specific learned feature from the previous layer. However dropout makes it so neurons cannot always rely on one specific neighbour, more the general inputs. This (hopefully) causes neurons to generalize better from all input. One can imagine dropout as running many many different variations of the same network.

This can help the model , but may (much like the deeper network) also increase overfitting (a regular risk when making a model more complex). 

[wide](imgs/hyperparams_MyMLPWide.png)

Generally speaking, both the train and validation losses are a little more "jagged", compared to the sometimes smoother loss lines we see in other models. We see the models learn, but certain neurons are not always learning, they have been turned off, so not all neurons are learning at the same pace.

### Best model. 

The best model is selected by valiadation accuracy. The model with the best valiadation accuracy was one of the baseline models. Below you can see it's performance:

[best_model](imgs/best_model.png)

It had a validation accuracy of **0.880**, but this is not extraordinary, as other models had very close validation accuracies, like one of the wide models which acheived a validation accuracy of **0.878**.

What is interesting is this model seems to overfit, as its valiadation loss is quite high, but as disucssed earlier, loss function =/= performance metric.

Lets see how well the model does on training data.

**Train data**

[conf_train](imgs/confusion_matrix_train.png)

Generally, the model manages to guess almost all planes as planes. It is a little worse on birds, sometimes guessing them as planes. But the model performs well.

**Validation data**

[conf_val](imgs/confusion_matrix_val.png)

The validation data tells a similar story to that of the training data, as it does well on planes, and a fair bit worse on birds, albeit a bit worse across the board.

With a performance metric like accuracy, a model may sometimes get away with guessing slightly more of one class than the other. 

Generally, the model manages to guess almost all planes as planes. It is a little worse on birds, sometimes guessing them as planes. But the model performs well.

**Test data**

[conf_test](imgs/confusion_matrix_test.png)

Again, a similar case for the test data, but now the model is worse on correctly predicting planes. The model generalizes, but not as well as on validation data. Perhaps the test data includes some particularly hard to spot images of birds/planes? Let's have a look.

**Incorrectly classified images**

Here are misclassified images in the test set and the model's confidence in its prediction.

[wrong_test](imgs/wrong_images_test.png)

Some images are clearly very "On the edge", like the very first image (index = 0) includes a bird, but the probability of a plane (0.54) was just slightly higher. So this image was a toss up. Other images are completely different, for instance the image with index = 3, the model is 100% certain is a plane, while we can see it is a close-up of a bird.

Closeups

Why is the model making mistakes? First off, the task is hard. Birds and planes are oftentimes in the sky, so using the blue background is not an option to distinugish them.
The specific model we chose actually had quite a high validation loss, and this can help explain the sometimes very wrong predictions.

### Did something go against expectations?

I had high hopes for the dropout model as from previous experience, these can be very effective. Perhaps given more data, time, layers or "wideness", these could perform well.

## Conclusion:

Generally, the models 