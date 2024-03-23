# Fitness Tracker Machine Learning Project

**Hello!**   I'm Ahmed TAIBECHE, an AI enthusiast currently learning and exploring the world of machine learning. Feel free to connect and see my profile!

**Project Goal**

This project aims to process, visualize, and model accelerometer and gyroscope data from a fitness tracker to create a machine learning model capable of classifying barbell exercises and counting repetitions.

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
