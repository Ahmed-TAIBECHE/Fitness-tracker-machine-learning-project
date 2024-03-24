# Fitness Tracker Machine Learning Project

**Hello!**   I'm Ahmed TAIBECHE, an AI enthusiast currently learning and exploring the world of machine learning. Feel free to connect and see my profile!

**Project Goal**

This project aims to process, visualize, and model accelerometer and gyroscope data from a fitness tracker to create a machine learning model capable of classifying barbell exercises and counting repetitions.


# Attention my goal from this project is to :

### 1. data processing and creating features from raw data

### 2.Experience with machine learning algorithms not to code.This project draws upon my experience with machine learning algorithms. I've implemented a variety of algorithms myself, while also incorporating elements from publicly available sources where applicable. These external sources are included at the beginning of relevant files within commented sections for reference. My original code contributions are also clearly marked within the files


**Acknowledgement**

A big thank you to mhoogen: `https://github.com/mhoogen/ML4QS/` ML4QS repository, an extension for the book "Machine learning for the quantified self. On the art of learning from sensory data" by Hoogendoorn, M., & Funk, B. (2018), for providing inspiration and code utilized in this project.

**Project Overview**

This project is structured into six parts:

* **Part 1:** Introduction, Goal, Quantified Self, MetaMotion Sensor, Dataset
* **Part 2:** Converting Raw Data, Reading CSV Files, Splitting Data, Cleaning
* **Part 3:** Visualizing Data, Plotting Time Series Data
* **Part 4:** Outlier Detection, Chauvenet's Criterion and IQR
* **Part 5:** Feature Engineering, Frequency, Low Pass Filter, PCA, Clustering
* **Part 6:** Predictive Modelling, Naive Bayes, SVMs, Random Forest, Neural Network
* **Part 7:** Counting Repetitions, Creating a Custom Algorithm


# You can run this project by running the next codes by order (which are totaly created by me ):
### 1. makes_dataset.py
### 2. visualize.py
### 3. remove_outliers.py
### 4. build_features.py
### 5. train_model.py
### 6. predict_model.py
### 6. count_repetitions.py



**Data**

The data used in this project was collected during gym workouts where five participants performed various barbell exercises using the MetaMotion sensor, which captures accelerometer and gyroscope readings.

**Running the Project**

**Important Note:** This project is designed for interactive execution within an IDE like Visual Studio Code. Read and execute each part of the code separately for better control and debugging.

**Option 1: Using pip**

1. Install required packages: `pip install -r requirements.txt`

**Option 2: Using conda**

1. Create a conda environment: `conda env create -f environment.yml`
2. Activate the environment: `source activate <environment_name>` (replace `<environment_name>` with the actual name)

**Further Information**

The project code is organized into separate scripts in the src folder (e.g., `makes_data.py` etc.) within the project directory. Explore and adapt the code for your own experimentation within your preferred interactive environment.
