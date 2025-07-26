A Step-by-Step Guide to Your Face Identification System
This document outlines the procedure for operating the face identification software, including training the model, managing the dataset, and running predictions.
1. Data Preprocessing: The Foundation of the Model
The initial and crucial step is to prepare the image data. This is handled by the preprocessing.py script located in the utils folder. Running this script presents you with several options to manage your dataset:
 Train from dataset (preprocess existing data): Select this option if you have an existing dataset of images that you want to use to train the model. This will prepare your data for the subsequent training phases.
 Collect test images from webcam only: This option allows you to capture new images using your webcam. These images can then be used to test the model's performance. After capturing, you will need to run the preprocessing script again to integrate this new data.
 Collect test images + Train model: This is a comprehensive option that first allows you to capture new images via webcam and then immediately preprocesses the entire dataset for training.
 Collect new dataset images from webcam: Use this option to add new individuals or more images of existing individuals to your core dataset.
Important: Every time you add new data to your dataset using options 2, 3, or 4, you must re-run the preprocessing script and select option 1 to ensure the new data is correctly formatted and included for training.
Upon successful preprocessing, the system will display the total number of training and testing samples, along with the labels for each class (individual). For example:
Generated code
##Train samples: 3241 | Test samples: 811
Labels: ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11']##
Use code with caution.
You will also receive a report on the number of frames per class and any frames where a face could not be detected. This is useful for identifying potential issues with the data for a specific class. If you find a high number of undetectable frames, you can use the edit_dataset.py script to clean up the data by removing poor-quality images.
The output of this preprocessing step are the files train_x and train_y, which contain the image data and corresponding labels, respectively. These files are essential for the subsequent stages.
2. Exploratory Data Analysis (EDA): Refining the Dataset
The next step is to run eda.py. This script performs a deeper analysis of your dataset to identify and remove redundancies and outliers, leading to a more robust and efficient model.
The EDA process includes:
Correlation Matrix: A visualization will be displayed showing the correlation between different classes. Blue areas indicate low correlation (good), while red areas suggest high similarity between classes, which could confuse the model. The script is set to remove data with a correlation of 0.99 to 1, but this threshold can be adjusted within the eda.py file.
Sample Image Review: You will be shown sample images that the model has identified from each class.
Outlier Removal: The script will identify potential outlier images that deviate significantly from the rest of the data within their class. You will be prompted in the terminal to enter the indexes of any outliers you wish to remove. If you don't want to remove any, you can simply press 'q' to quit.
Class Variance Check: This analysis helps to identify classes with very low variance. Low variance might indicate that the images for a particular individual are too similar, which could lead to the model being biased and performing poorly on new, slightly different images of that person.
3. Feature Extraction: Translating Faces into Numbers
After cleaning the data with EDA, you need to run feature_extraction.py from the interpretability folder. This script uses the Local Binary Patterns (LBP) algorithm.
LBP is a texture descriptor that analyzes the relationship between a pixel and its surrounding neighbors. In face identification, LBP is effective at capturing the fine-grained texture of the skin, such as wrinkles and pores, and converting these features into a numerical vector. This vector representation is what the machine learning model will use to learn the unique characteristics of each face. A threshold, in this case, set to 60, is used during this process to fine-tune the feature extraction.
4. Model Training: Teaching the System to Recognize Faces
The penultimate step is to train the classifier by running train_classifier.py. This script employs the K-Nearest Neighbors (KNN) algorithm.
KNN is a simple yet powerful classification algorithm. When given a new, unknown face, it looks at the 'K' most similar faces from its training data. The new face is then assigned to the class that is most common among its 'K' nearest neighbors. The training process involves repeatedly testing the model against the test data until the confusion matrix, a table showing the model's prediction accuracy for each class, is satisfactory. The goal is to minimize the instances where one class is incorrectly identified as another.
The accuracy of the model is measured on a scale from 0 to 1. An accuracy of 0.98 indicates that the model is making correct predictions 98% of the time on the test data.
5. Prediction: Putting the Model to the Test
Finally, you can interact with the trained model through the GUI.py script.
Before you can start making predictions, you must first click the "Train the model" button within the application. This loads the trained model. Once the model is loaded, you have two ways to test its face identification capabilities:
Live Webcam: The system can access your webcam to perform real-time face identification.
Image Upload: You can select a JPG image file from your computer for the model to predict the identity of the person in the photo.
