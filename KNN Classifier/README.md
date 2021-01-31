# KNN Classifier 
- Specifications:
1. Train folder that contains folders*, each folder will contain the dataset to train the model.
2. Test folder that contains folders*, each folder will contain the dataset to train the model.

- Preprocess:

>Each sub folder in the train/test set goes through the same preprocess steps. 
>1) converting to gray scale
>2) if the width != height, add padding to the small one (white border) to make the picture a square so that width = height
>3) resizing to 40X40

>The program used the ration of 90:10 for the train/validation sets

- Train:

>Giving the model the preprocessed dataset and train it using:
>1. Chi Square distance function
>2. Euclidean distance function

>The program will take the best trained model with K neighbor from 1 to 15 (usually the best is between 7 to 9).

- Test:

>After training will test the best model using the preprocessed test dataset.

- Result:

>Will save a csv file that contains confusion matrix and classification report.
>Can be changed to save the best model.

	*folders = letter folders (0 = a, 1 = b and so on).
	Example of how the file tree should look like:
		-root
			- Train
				- 0
					- pic1.jpg
					- pic2.jpg
					- pic3.jpg
				- 1
				- 2
				...
			- Test
				- 0
				- 1
					- pic1.jpg
					- pic2.jpg
					- pic3.jpg
				- 2
				...
				
			- preprocessed.py
			- knn_classifier.py

# Installation
python version 3.7

libraries needed: opencv, numpy, os, argparse, pandas, scikit-learn, scikit-image, datetime, random

# How to use?
>1. open terminal/CMD
>2. pre process TRAIN and TEST folders -> python preprocess.py [--train TRAIN_PATH] [--test TEST_PATH].
>
>	2.1. it will be output to "processed_train" or "processed_test".
>3. start the classifier -> python knn_classifier.py <path to train set> <path to test set>.
>	
>	3.1. output -> csv file that will hold both knn distance results.
>
>*Note: can change the code to save the model and use it in various ways.
	
# Credits
>Aviel Cohen - [github] (https://github.com/AvielCo)
>[linkedin](https://www.linkedin.com/in/aviel-cohen-a5840216b/)
>
>Noah Solomon - [github](https://github.com/SoloNoah),
>[linkedin](https://www.linkedin.com/in/noah-solomon-b40573135/)
>
>Emilia Zorin - [github](https://github.com/EmiliaZorin),
>[linkedin](https://www.linkedin.com/in/emilia-zorin-417635168/)
>
>Vitaly Nechayuk - [github](https://github.com/VitNecha),
>[linkedin](https://www.linkedin.com/in/vitaly-nechayuk/)

# Dataset contributors

>The dataset used in this project was HHD_v0
>HHD_v0 was divided into TEST/TRAIN sets which included all 26 sub directories related to each hebrew letter.
>The dataset, HHD_v0, was developed by  I. Rabaev, B. KurarBarakat, A. Churkin and J. El-Sana for their paper [HHD_v0](https://www.researchgate.net/publication/343880780_The_HHD_Dataset)


