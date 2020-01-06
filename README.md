# Disaster-Response-Project
Project in Data Scientist Nanodegree of Udacity - Message Classification 

### Table of Contents

1. [Installation](#installation)
2. [Project Overview](#overview)
3. [File Descriptions](#files)
4. [Implementation](#implementation)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>
This is a web application project built by HTML and Python version 3.* while it contains many python pachages such as pandas, ntlk, flask, numpy, re, pickle and so many. 

## Project Overview<a name="overview"></a>

This project classifies disaster messages into 36 categories. This is a web project built by HTML and Python. Hence, it allows user ti enter a message and then the message get classified among the 36 categories, which indicates the disaster. 

## File Descriptions <a name="files"></a>

* **app**: It is a folder that contains the HTML page and python file to run the app.
* **data**: It is a folder that contains the learning data which are messages and categories, besides its processing functions that implemented by python. 
* **models**: It is a folder that contains the learning algorithm implementation.
* **ETL Pipeline Preparation**: It is a Jupyter notebook that illustrates the data processes this project is done.
* **ML Pipeline Preparation**: It is a Jupyter notebook that illustrates the machine learning algorithm is used by this project.

## Implementation Instructions<a name="implementation"></a>

In order to run the project after download the workspace, there are three steps must be taken:

### First: Process Data.
	1. From the IDE Navigate to data.
	2. Run "python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db"

### Second: Create Learning Model.
	1. From the IDE Navigate to models.
	2. Run "python train_classifier.py ../data/DisasterResponse.db classifier.pkl"
	
### Third: Run the Web App.
	1. From the IDE Navigate to app.
	2. Run "python run.py"
	3. Go to http://0.0.0.0:3001/

## Screenshots

***Screenshot 1: App Front Page***
![Screenshot 1](https://github.com/Sultan660/Disaster-Response-Project/blob/master/screenshot1.PNG)

***Screenshot 2: App Results Page***
![Screenshot 2](https://github.com/Sultan660/Disaster-Response-Project/blob/master/screenshot2.PNG)

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

Licensing for the data is issued by Udacity and this project is made to fulfill an nanodegree requirement.