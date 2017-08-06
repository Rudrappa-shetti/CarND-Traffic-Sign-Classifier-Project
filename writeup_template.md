#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

All output bar charts in the folder : CarND-Traffic-Sign-Classifier-Project\test_pics

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because grayscaling would make the neural network faster.
TensorFlow's tf.image.rgb_to_grayscale() function was an option, but I found that to be a little slow. 
OpenCV's grayscaling was also an option, although it gives back only height and depth, and would therefore require some
additional work on the model to get it to feed in correctly. Numpy's "newaxis" feature allowed me to add back the 
additional 'depth' of 1 so that I did not need to redo my neural network further.
I normalized the data between .1 and .9, similar to what was discussed in one of the TensorFlow lectures, 
so that I could keep numerical stability for any of the larger mathematical sequences occurring. This helps scale 
down some of the disparity within the data. Also, using .1 to .9 avoids any potential problems incurred by allowing 
the data down to zero (which could potentially break some equations down the road)


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers: Totally 5 layes - 2 Convolution and 3 Fully connected Network

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale image   					| 
| Convolution layer1    | Input = 32x32x1. Output = 28x28x6. 			|
| RELU					|												|
| Average pooling	    | Input = 28x28x6. Output = 14x14x6 			|
| Convolution layer2	| Output = 10x10x16.     						|
| RELU					|												|
| Average Pooling		| Input = 10x10x16. Output = 5x5x16				|
| Flattening			| Input = 5x5x16. Output = 400.					|
| Fully connected1		| Input = 400. Output = 120						|
| RELU					|												|
| Dropout				| to prevent overfitting						|
| Fully Connected2      | Input = 120. Output = 84.						|
| RELU					|												|
| Dropout				|												|
| Fully Connected3      | Input = 84. Output = 43.						|
| Softmax				| 		       									|
|						|												|
|						|												|
 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model,  I utilized the AdamOptimizer from within TensorFLow to optimize, which seemed to do better than a regular Gradient Descent Optimizer.
Also, I tried a few different batch sizes (see below), but settled at 250 as that seemed to perform better than batch sizes larger or smaller than that. 
I ran only 20 epochs, primarily as a result of time and further performance gains, as it was already arriving at nearly 99% validation accuracy, 
and further epochs resulted in only marginal gains while continuing to increase time incurred in training. 
Additionally, there is no guarantee that further improvement in validation accuracy does anything other than just overfit the data (although adding dropout to the model does help in that regard).

For the model hyperparameters, I stuck with a mean of 0 and standard deviation/sigma of 0.1. 
An important aspect of the model is trying to keep a mean of 0 and equal variance, so these hyperparameters attempt to follow this philosophy.
 I tried a few other standard deviations but found a smaller one did not really help, while a larger one vastly increased the training time necessary.

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 93.6
* validation set accuracy of 99.0% 
* test set accuracy of internet images is 83%.

My network is a convolutional neural network, as these tend to do very well with images.
I mostly used the same architecture as the LeNet neural network did, with 2 convolutional layers and 3 fully connected layers. 
I also did a few attempts with one less convolutional layer (which sped it up by a decent amount but dropped the accuracy) 
as well as one less fully connected layer (which only marginally dropped the accuracy).

One item I did change from the basic LeNet structure was adding dropout to the fully connected layers. 
Although this makes initial epochs in validation a little worse, I gained an additional 3% on test accuracy. 
Since I was getting to validation accuracy of around 99%, with test accuracy 93%. Dropout helped preventing some of overfitting. 
I put dropout at 0.7 probability as that tended to still validate at a decent rate within an acceptable number of epochs over a lower number such as 0.5. 
Also, I switched max pool to average pool as that seemed to slightly increase accuracy.
Note also that due to grayscaling, and adding back a dimension (see above) for depth, the inputs for my network need to be at 32x32x1, instead of the original 32x32x3 that the data had at first.
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

all the images are present in the test images folder. : CarND-Traffic-Sign-Classifier-Project\test_pics

All images easier to classify.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Priority road         | Priority Road    							    | 
| Stop Sign     		| Stop Sign 									|
| No Entry				| No Entry										|
| general caution	    | general caution					 			|
| 60 speed				| Right of way at next intersection  			|
| roundabout mandotory  | roundabout mandotory							|	


The model was able to correctly guess 5 of the 6 traffic signs, which gives an accuracy of 83%. This compares favorably to the accuracy on the test set of 93%

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 17th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a Priority road (probability of 1.0), and the image does contain a Priority road. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Priority Road   								| 
| 1.0     				| Stop Sign 									|
| 1.0					| No Entry										|
| 1.0	      			| general caution					 			|
| 1.0				    | Right of way at next intersection       		|
| 1.0					| roundabout mandotory							|


For the second image, the model is relatively sure that this is a stop sign (probability of 1.0), and the image does contain a stop sign. 
For the third image,  the model is relatively sure that this is a No Entry (probability of 1.0), and the image does contain a No Entry.
For the fourth image, the model is relatively sure that this is a general caution (probability of 1.0), and the image does contain a general caution.
For the fifth image, the model is relatively sure that this is a Right of way at next intersection (probability of 1.0), and the image does not contain a Right of way at next intersection.(here the prediction went wrong)
For the sixth image, the model is relatively sure that this is a roundabout mandotory (probability of 1.0), and the image does contain a roundabout mandotory.




