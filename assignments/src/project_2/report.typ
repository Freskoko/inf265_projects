#set page(
  paper: "a4",
  margin: (x: 2cm, y: 2.5cm),
)
#set text(
  size: 11pt,
)
#set heading(numbering: "1.")

= INF265: Project 2: CNN

== Object localization:

=== Data exploration, preprocessing

We initially normalized the data, then plotted some of the images with their items and bounding boxes:

#figure(
  image("imgs/object_localization/data_explore_train.png", width: 60%),
  caption: [Train data exploration],
)

We then had a look at the class distribution.

#figure(
  image("imgs/object_localization/class_dist.png", width: 60%),
  caption: [Class distribution],
)

Seems like classes are mostly balanced, looks good.

We then plotted how many images have an item vs no item.

#figure(
  image("imgs/object_localization/class_dist_items.png", width: 60%),
  caption: [Item vs No Item distribution],
)

Seems like the distribution is unequal, the model could get away with guessing that there is an item in the image when there actually is none, so we should keep track of how good the model is at guessing if there is an item in the image or not.

Lets look at the pixel value distribution

#figure(
  image("imgs/object_localization/pixel_dist.png", width: 60%),
  caption: [Pixel distribution],
)

Looks normal, with peaks at -2 and ~2.7.

Next up, we had a look at the pixel values outside/inside the bounding boxes.

#figure(
  image("imgs/object_localization/inside_outside_pixel_dist_pre_process.png", width: 60%),
  caption: [Pixel distribution pre-process],
)

Seems like the pixels we are interested in tend towards higher values.
We tried to de-noise the data by setting all pixels under a value of 1.5 to 0.

#figure(
  image("imgs/object_localization/inside_outside_pixel_dist_post_process.png", width: 60%),
  caption: [Pixel distribution post-process],
)

Now there is a clear divide!

#figure(
  image("imgs/object_localization/data_explore_train_post_process.png", width: 60%),
  caption: [Post-process training data],
)

And the images still look ok. Nice, hopefully this helps the model.

=== Models

*Loss function*

Before we get into the model, we have to talk about the loss function.

Our loss function is *Localization loss* for the image. This loss function consitsts of three parts.

1. Detection loss, "is there an item in the image?"
2. Localization loss, "Where is the item in the image"?
3. Classfication loss, "What class is in the image"?

We chose to have all three to have equal weight. In hindsight however there could be some merit to weighting "where is the item in the image" as a more important component, since this is what the model seemed to struggle on.

However, if there is no item in the image, we instead only use the Detection loss.

*Performance metric*

Models were evaluated using class accuracy and IoU as the main performance measures, taking their mean. From now on if this report mentions "accuracy", it means the mean of class accuracy and IoU.
We also included object accuracy as a secondary metric, but almost all models very quickly were perfect when guessing if there was an item in the image or not.

=== Selected models

For object localization, we tried a wide range of models to see how they performed.

#table(
  columns: (auto, 1fr),
  inset: 10pt,
  align: horizon,
  [*Model*], [*Description*],
  [CNNBaselineNoBatch], [Basic CNN without BatchNorm],
  [CNNBaselineWithBatch], [CNN with BatchNorm],
  [CNNDeep], [Deeper CNN + Batchnorm],
  [CNNResNet], [CNN with residual connections + Batchnorm],
  [CNNDenseNet], [CNN with dense connections + Batchnorm],
)

Since there is only one object per image, the models use fully connected layers for the final prediction.
In this last fully connected layer we added dropout for regularization.

The following hyperparameters were selected:

#table(
  columns: (auto, auto),
  inset: 10pt,
  [*Hyperparameter*], [*Values*],
  [Batch size], [32],
  [N epochs], [15],
  [Learning rate], [_1e-3, 1e-4_],
  [Weight decay], [_0, 1e-4_],
  [Dropout], [_0, 0.3_],
)

A grid search over learning rate, weight decay and dropout was used during training of each model. Batch size and number of epochs were kept constant across all trainings.

== Models and performances

=== Baseline model

This baseline model consists of 3 convolutional layers into maxpool layers, and a fully connected layer (with dropout), and the output layer. Kernel sizes are 3\*3, with padding 1 and stride 1, keeping the dimensions the same, and the maxpool is responisbile for reducing dimension size. This model does NOT have batch normalization.

Generally, these baseline models perform decently well, but some of them overfit. The 4 models in the top of the graph shows signs of overfitting, as validation loss increases while train loss decreases. The bottom four show a different curve, where train and validation loss decrease steadily together. The top models have the same learning rate (0.001) and this is why we see overfit there, while hte bottom models have a low learning rate (0.0001), which is why they are taking a long time to learn. It may seem like the bottom 4 models are better, but looking at validation accuracy and the loss, the top four models are performing generally better.

Generally, these models acheive a good class and object accuracy, but their *IoU* varies quite a bit, some models getting 0.41, while others acheving IoU's of 0.59.

#figure(
  image("imgs/object_localization/hyperparams_CNNBaselineNoBatch.png", height: 90%),
  caption: [NoBatch Baseline Results],
)

=== Baseline + batch norm

This baseline model consists of 3 convolutional layers into maxpool layers, and a fully connected layer (with dropout), and the output layer. Kernel sizes are 3\*3, with padding 1 and stride 1, keeping the dimensions the same, and the maxpool is responisbile for reducing dimension size. This model DOES have batch normalization.

These models are very similar to the other basline above, but do generally slighly better, seeing just slight improvements in validation score. This shows the effect that batch norm has, in creating more stable updates for the model.

#figure(
  image("imgs/object_localization/hyperparams_CNNBaselineWithBatch.png", height: 90%),
  caption: [Batch Norm Baseline Results],
)

=== CNNDeep

This model is very similar to the two baselines, but instead of 3, it has *4* convolutional layers. This model is more complex, and can hopefully learn more complex/abstract features from images.

There are some interesting models here, most notably the bottom right one, which managed to have a lower validation loss than training loss for any epochs, until about epoch 8 where validation loss is lower than training loss.
For these models, we again see a slight improvement in validation accurcacy.
These models, much like earlier ones, are really good at determining if there is an item in the image, and what class it is, but getting any IoU over 0.60 seems difficult.

#figure(
  image("imgs/object_localization/hyperparams_CNNDeep.png", height: 90%),
  caption: [CNNDeep Results],
)

=== CNNDenseNet

A densenet is a kind of model that uses dense blocks/connections to give the input of previous layers into the future layers. So all features learned in early layers are also given to later layers. The idea is that this preserves spatial information.

Generally, the models perform quite well, but their validaton losses are somewhat erratic.

#figure(
  image("imgs/object_localization/hyperparams_CNNDenseNet.png", height: 90%),
  caption: [CNNDenseNet Results],
)

=== CNNResNet

A resnet is a kind of model that adds the original input of the image back to the output of layers. This is a typical model for image related tasks, so we expected it to perform well.

Generally, the models perform quite well (validation accuracy around 0.70 at 15 epochs), with the bottom right model even reaching a validation accuracy of 0.75.
Interestingly, these models do quite well on IoU, but their class accuracy is a little bit lower than that of the baseline models. These models manage to learn and perform so well becuase they preserve spatial information across multiple layers and learn complex features.

Most models are doing quite well even after 15 epochs, and it would be interesting to perhaps train some of them for a longer period of time. Certain models however are obviously overfitting (bottom 2), but their validation accuracies are still increasing. An interesting example of the fact that loss function =/= performance metric

Interestingly, the bottom four models seem to have an erratic validation loss, jumping up and down. These are the four models with a low learning rate (0.0001), but after 15 epochs, they have much the same validation scores as the models with a higher learning rate (0.001).

#figure(
  image("imgs/object_localization/hyperparams_CNNResNet.png", height: 90%),
  caption: [CNNResNet Results],
)

=== Best model:

the best model was CNNResNet with these hyperparameters. It was chosen based on validation accuracy score.

#table(
  columns: (auto, auto),
  inset: 10pt,
  [*Hyperparameter*], [*Values*],
  [Learning rate], [1e-4],
  [Weight decay], [1e-4],
  [Dropout], [0.3],
)

Here is a graph showing its performance:

#figure(
  image("imgs/object_localization/best_model.png", width: 80%),
  caption: [Best localization model performance],
)

It's object accuracy is high, but this is the same for all models, the interesting thing is its *IoU* and *class accuracy score*.
A validation average IoU score of 0.62 is quite high for a difficult task like this, and shows how the model can really perform. It's class accuracy is quite good, almost perfect. Looking at its "mean accuracy", which is our main performance metric, we can see that the class accuracy is dragging the mean accuracy up, and the iou is dragging it down.

There is some sign of overfitting as validation and train losses are diverging, but the mean accuracy performance metric is increasing, so it is not a negative, perhaps given more epochs the validation mena accuracy would fall.

=== Performance of the best model

Lets look deeper into how the best model performs. Now that we have selected it, we can see its performance on test data.

#figure(
  image("imgs/object_localization/final_accuracy_scores.png", width: 80%),
  caption: [Final accuracy scores],
)

The model getsa very similary mean accuracy score across all datasets, with a slight decrease in performance on test/validation data. A decrease in test data performance is expected, but only a slight decrease shows that the model is generalizing well.

=== Object confusion matrix

Lets look at an object confusion matrix, firstly on training data.

#figure(
  image("imgs/object_localization/object_confusion_matrix_train_train.png", width: 60%),
  caption: [Confusion Matrix Train],
)

And then on test data.

#figure(
  image("imgs/object_localization/object_confusion_matrix_test_train.png", width: 60%),
  caption: [Confusion Matrix Test],
)

The model is generally quite good at understanding when there is an item in the image.

=== Bounding boxes

Lets look deeper into the bounding boxes. One way to look at this is "bounding box RMSE"

Firstly on train.
#figure(image("imgs/object_localization/bb_rmse_train.png", width: 60%))

Then on validation
#figure(image("imgs/object_localization/bb_rmse_val.png", width: 60%))

And then on test.
#figure(image("imgs/object_localization/bb_rmse_test.png", width: 60%))

Seems like across train, validation and test, our RMSE for the images of the letter 1 and 4 are a fair bit higher, perhaps these bounding boxes are generally harder to predict?
The scores (0.05) may seem really low, but considering we are working with normalized data, this is actually fairly high.
Otherwise, all classes see a similar bounding box rmse.

Next, let us view some actual predictions.

Firstly on validation data.

#figure(
  image("imgs/object_localization/data_predict_val.png", width: 80%),
  caption: [Predictions on validation data],
)

#figure(
  image("imgs/object_localization/data_predict_test.png", width: 80%),
  caption: [Predictions on test data],
)

Looks like our model is managing to predict the bounding boxes for test data quite well!
They are not perfect, but they manage to somewhat capture the items.

= Object detection:

== Approach & Design choices

=== Data exploration:

NOTE: *We did not use the pre-processed bounding box data, we implemented the `global_to_local` and `local_to_global` functions.*

Initially, we normalized the data, and explored it by plotting some images from the training data. Here are images with index 0-11 in the training data.

#figure(image("imgs/object_detection/data_explore_train.png", width: 60%))

The images look different enough, a model could for sure learn the different labels. Each of the bouding boxes also seem appropriate.

Lets check the class distribution:

#figure(image("imgs/object_detection/class_dist.png", width: 60%))

Hmm, there are slightly more "number 1's" in the images than 0's. The difference is relativley small between them, so this may not be a big issue.

Next, lets plot the pixel distribution of the images

#figure(image("imgs/object_detection/pixel_dist_pre.png", width: 60%))

Strange, seems like there is a very high pixel value, even after normalizing.
Lets find it and plot it.

#figure(image("imgs/object_detection/outlier.png", width: 60%))

Ok so we have an outlier at index 24487. What a strange image.
Considering only one image has this very weird pixel with a value of 5.68 (very large), we can delete it.
This should not affect model performance to a large degree, as we are deleting one image which (seemingly) has a very strange pixel value.

#figure(image("imgs/object_detection/pixel_dist.png", width: 60%))

After deleting that image, the pixel distribution is nice and normal, and there are peaks at 0 and 1. Seems like very white and very black pixels are common in these images. Nothing weird here.

Let's have a look at pixel values outside / inside bounding boxes.

#figure(image("imgs/object_detection/inside_outside_pixel_dist_pre_process.png", width: 60%))

Seems like the pixels we are interested in (light green) tend towards higher values.
We tried to de-noise the data by setting all pixels under a value of 1.5 to 0.

#figure(image("imgs/object_detection/inside_outside_pixel_dist_post_process.png", width: 60%))

Now there is a clear divide!

Lets look at some images to see the effect.

#figure(image("imgs/object_detection/data_explore_train_post_process.png", width: 60%))

And the images still look ok. Nice, hopefully this helps the model.

=== Models

*Loss function*

Before we get into the model, we have to talk about the loss function.

We perform *Localization loss* for all 6 cells (2\*3 = 6). In this way the model is trained to evaluate each cell.

*Performance metric*

Models were evaluated using MAP (mean avergae precision). We also added IoU, object accuracy and classification accuracy as additional metrics to view model performance, but MAP score was what decided which model we picked as the "best one"
MAP is chosen in this task because we can theoretically have "infinite TRUE NEGATIVES", so we need a performance metric that does not use true negatives. We would use recall here awswell, but this could be abused by the model, as it may start to place many more boxes than required.

=== Selected models

For object detection, the following models have been used:

#table(
  columns: (auto, 1fr),
  inset: 10pt,
  [*Model*], [*Description*],
  [CNNBaselineNoBatch], [Basic CNN without BatchNorm],
  [CNNBaselineWithBatch], [CNN with BatchNorm],
  [CNNLargeKernels], [CNN with larger convolution kernels],
  [CNNResNet], [CNN with residual connections],
  [CNNDenseNet], [CNN with dense blocks],
)

Generally, the models consits of mulitple sets of:

Convolutional layer -> batch normalization -> maxpool

These models were fully convolutional considering the nature of the detection problem.

Models were trained with the following hyperparameters

#table(
columns: (auto, auto),
inset: 10pt,
[Hyperparameter], [Values],
[Batch size], [32],
[N epochs], [15],
[Learning rate], [1e-3, 1e-4],
[Weight decay], [0, 1e-2, 1e-4],
)

The same grid search over learning rate and weight decay was used during training, and batch size and number of epochs are constant across trainings.
Below are plots over all model performances:

== Models and performances

=== Baseline model

Please excuse the spelling mistake in the graph, "Basline" should be "Baseline"

This baseline model consists of 4 convolutional layers into maxpool layers (and a final output convolutional layer). In order to turn our 48 x 60 input into the required 2x3, we use an uneven kernel size (3x2) in the fourth layer. This model does NOT have batch normalization.

Generally, these baseline models perform well, and some of them may do better, if given more epochs.
One of the interesting grapphs is the top right model, which has seemingly better validation loss than training loss!
Another interesting model are the models on the bottom, where the only difference is the learning rate, and one can see that the difference between the two is very small. However, the model with the slower learning rate performs slightly better on training data.
Generally, these models acheive a good class and object accuracy, but their IoU is quite quite poor.

#figure(image("imgs/object_detection/hyperparams_CNNBaslineNoBatch.png", height: 90%))

=== Baseline + batch norm

This baseline model consists of 4 convolutional layers into maxpool layers (and a final output convolutional layer). In order to turn our 48 x 60 input into the required 2x3, we use an uneven kernel size (3x2) in the fourth layer. This model DOES have batch normalization.

Already from adding batch normalization we can see these models are performing better than our baseline without batch normalization (about an average +10 validation map score at epoch 15). Batch normalization stablizes features distributions in a way that improves training, even without resdidual connections (which we will look at later).
Generally, these models see stable learnng updates, and models have generally higher validation IoU. The models with higher validation IoU, tend to have a higher validation map score.
A notable model here is the model with learning_rate: 0.001, weight_decay: 0.0001, (row 2, image 1). This model sees a sterk improvement in iou and validation accuracy over epochs. It may be interesting to run this model for more epochs, as it seems the model is still improving. This may however lead to overfit.

#figure(image("imgs/object_detection/hyperparams_CNNBaslineWithBatch.png", height: 90%))

=== CNNLargeKernels

This is a similar model to the baselines, but it has quite large kernel sizes

#table(
columns: (auto, auto),
inset: 10pt,
[Layer], [Kernel size],
[1], [7],
[2], [5],
[3], [3],
[4], [3],
)

Increasing kernel sizes increases the receptive field of the neurons, allowing each of them to "see more" of the image.
This may allow the neurons to learn larger spatial features, gathering broad information.
However, as the graphs below show, this model underperforms quite a bit. This model is even worse than the baseline without batch normalization.
Most of the models seem to learn quickly in the start, but then after about 5 epochs, the loss drops very slowly.
The two bottom models seem to be training nicley, and could perhaps improve given more epochs.
Looking back, using large kernels + pooling may not be the best for this task, as information is perhaps lost too quickly. It was still an interesting model to try, and now we understand why this architechure is poor for this problem.

#figure(image("imgs/object_detection/hyperparams_CNNLargeKernels.png", height: 90%))

=== CNNResNet

A resnet is a kind of model that adds the original input of the image back to the output of layers. This is a typical model for image related tasks, so we expected it to perform well.

Generally, the models perform quite well (validation map around 0.50 at 15 epochs), with the exception of the top right model.
It has params learning_rate: 0.001, weight_decay: 0.01. It's learning seems to stagnate after 6 epochs, learning very slowly.
These models manage to learn and perform so well becuase they preserve spatial information across multiple layers and learn complex features, but can also localize the objects (since they have the input image).

Most models seem to begin to overfit after about 15 epochs, so perhaps this was a good amount of training time for them. For example, the bottom right model, which has a training loss of almost 0 at the end.

#figure(image("imgs/object_detection/hyperparams_CNNResNet.png", height: 90%))

=== CNNDenseNet

A densenet is a kind of model that uses dense blocks/connections to give the input of previous layers into the future layers. So all features learned in early layers are also given to later layers. The idea is that this preserves spatial information.
These models seem to perform ok, somewhat better than the baseline with batch norm, but worse than the ResNet.
Most model learning stagnates after a while, and learning is slow. Validation loss is very closely tied to training loss, meaning that the model is generalizing well.
Interestingly, the two models on the top of the graph differ a lot in performance, but their only difference is that one has weight_decay 0, and the other has weight_decay 0.01. The model with weight decay 0 gets a validation map score of 41, while the other gets a validation map score of 19.

#figure(image("imgs/object_detection/hyperparams_CNNDenseNet.png", height: 90%))

=== Best model:

the best model was CNNResNet with these hyperparameters. It was chosen based on "map" score.

#table(
columns: (auto, auto),
inset: 10pt,
[Hyperparameter], [Values],
[Learning rate], [1e-4],
[Weight decay], [1e-4],
[Dropout], [0.3],
)

Here is its performance:

#figure(image("imgs/object_detection/best_model.png", width: 80%))

This model does quite well.
It's class and object accuracy is high, but this is the same for all models, the interesting thing is its IoU score.
A validation average IoU score of 0.78 is quite high for a difficult task like this, and shows how the model can really perform.
This models is somehwta strange as its validation loss goes down slowly, but the validation loss is much more erratic.

=== Performance of the best model

Lets look deeper into how the best model performs. Now that we have selected it, we can see its performance on test data.

#figure(image("imgs/object_detection/final_map_scores.png", width: 80%))

The model getsa very similary MAP score across all datasets, with a slight decrease in performance on test data. A decrease in test data performance is expected, but this slight decrease shows that the model is generalizing well.

To understand the model closer, we can look at map_50 and map_75

#figure(image("imgs/object_detection/final_map_50_scores.png", width: 80%))

Looks like the model does very well here. Map_50 is easier since it is easier to label a bounding box prediction as "correct".

#figure(image("imgs/object_detection/final_map_75_scores.png", width: 80%))

Map 75 is much the story as the total map score, getting 75% is much harder

=== Class confusion matrix

Lets look at a class confusion matrix, firstly on training data.

#figure(image("imgs/object_detection/class_confusion_matrix_train_train.png", width: 60%))

And then on test data.

#figure(image("imgs/object_detection/class_confusion_matrix_test_test.png", width: 60%))

Seems like the model is quite good at understanding differences between 0/1 in an image. Both in training and test data. We already knew this from our graph, and all models quickly understood this.

Lets look deeper into the bounding boxes. One way to look at this is "bounding box RMSE"

Firstly on train.
#figure(image("imgs/object_detection/bb_rmse_train.png", width: 60%))

And then on test.

#figure(image("imgs/object_detection/bb_rmse_test.png", width: 60%))

Seems like for both train and test, our RMSE for the images of the letter 1 are a fair bit higher, perhaps these bounding boxes are generally harder to predict?
This score may seem really low, but considering we are working with normalized data, this is actually fairly high.

Let us view some actual predictions

Firstly lets see ALL box predictions on validation data:

Here we can see ALL the boxes predicted by our model. Ok seems to work.

#figure(image("imgs/object_detection/data_predict_val_all_imgs_boxes.png", width: 80%))

Lets add non-max-suppression:

There is no difference, as none of the boxes are really overlapping.

#figure(image("imgs/object_detection/data_predict_val_all_imgs_boxes_iou.png", width: 80%))

Lets add non max suppression AND only pick boxes with a confidence higher than 0.5.

#figure(image("imgs/object_detection/data_predict_val_all_imgs_boxes_confidence.png", width: 80%))

Looks like the model doing quite well, but struggling on some images. The boxes could be in better spots.

Finally, lets have a look at the test data with non-max suppression and only showing boxes with a confidence score higher than 0.5

#figure(image("imgs/object_detection/data_predict_test_all_imgs_boxes_confidence.png", width: 80%))

Looks like our model is performing alright!

Sometimes our bounding box is not really covering the right number, but generally our model manages to understand the class, and ish where the bounding box should be. Much like the test data, sometimes it is completely off.

=== Given more time

Given more time, it would be interesting to train certain models for more epochs (not all, as some were overfitting). It would also be interesting to try some sort of data augmentation, to create new images.

== Conclusion:

Generally, the all models perform well, but vary depending on the parameters chosen. Certain models overfit, others struggle to learn.
Overall, for both the localization and detection tasks, many of the numbers within the images were located and correctly classified.

Interestingly, non-max suppression did not matter so much for object detection, as the best model did not seem to create so many overlapping boxes.

The model that worked the best on both tasks, was a resnet model, where both of them has the same weight decay, but differed by learning rate. The object localization model has a learning rate of 0.0001, while the object detection resnet model had a learning rate of 0.001.

== On the use of AI

AI was used in this project, to assist in bug-fixing, plot creation, and understanding of the learning material.
AI has been cited in the code where appropriate.
In the format uib wishes: The service ChatGPT has been used to generate code for plotting and debugging. ChatGPT was also used to inquire into the differences between loss function and performance measure in terms of this assignment.

== Divison of labour

Henrik Brøgger did the code for object_detection + object_localization. While Tobias Skodven and Henrik Brøgger both worked on the report.
