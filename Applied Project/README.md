# Code sources for the model
**Speaker Recognition** - https://keras.io/examples/audio/speaker_recognition_using_cnn/#data-preparation

**Speech Recognition** - Python Library (called SpeachRecognition)

**Sentiment Analysis** - https://realpython.com/sentiment-analysis-python/#using-machine-learning-classifiers-to-predict-sentiment

# 1. INTRODUCTION

This is the final report of the project: build a speaker recognition and mood analysis system. The purpose of the system is to find a speaker in a given dataset and performing text-independent mood analysis. We employed AWS services, JSON, Python (an open-source programming language), and other tools. Initially, we were planning to use a couple of AWS services for machine learning, but we decided to use Colab after all. 

We used Amazon EC2 to build our server. We uploaded our website and the model with some audio files to the server to run the system. 

### FFT
Fast Fourier Transform (FFT) is used for speaker recognition and the result of each speaker shows the difference in the frequency-domain representation. The accuracy in both speaker recognition and mood analysis is high 0.95% and 0.89% respectively. 

### Limitations
* Only the five speakers: Benjamin Netanyahu, Jens Stoltenberg, Julia Gillard, Margaret Thatcher, and Nelson Mandela
* Only 30-second .wav format audio files are available 
* Available emotions : Positive and Negative

### Problem 
* The model for speaker recognition is trained with 1-second audio files and when it comes the different length of audio files, the model often fails to predict the correct speaker.
* The sound of the audio files converted from video files is not equal. The results of FFT are not equal even though the speaker is the same. This leads the model to the wrong prediction. 

# 1. Model

## Spekaer Recognition Model

### 1. Question Definition

Determine the five different speakers from a given audio file. 

### 2. Data Collection

* Kaggle - https://www.kaggle.com/kongaevans/speaker-recognition-dataset

* Youtube

<p align="center">
  <img width="760" height="563" src="https://user-images.githubusercontent.com/42757351/129858450-b334f058-f995-40ac-871a-f4a712a38d49.png">
</p>

* audiocheck (noise files)

<p align="center">
  <img width="315" height="130" src="https://user-images.githubusercontent.com/42757351/129858463-00d3080a-9e1c-41a0-91e7-d5eb4267ed24.png">
</p>

### 3. Data PreProcessing

Once the video files are collected, they are converted into .wav files by an video editor programme called Movavi.

Then, split them into 1-second long files.

* Number of audio files for each speaker 

<p align="center">
  <img width="500" height="520" src="https://user-images.githubusercontent.com/42757351/129861462-71e9d969-c175-48dd-b1b7-12b295296bf1.png">
</p>

### 4. Model Building

After that, train the model.

<p align="center">
  <img width="1500" height="510" src="https://user-images.githubusercontent.com/42757351/129880152-759f855f-2549-46e3-9d61-7f17b5dd4ff3.png">
</p>


<p align="center">
  <img width="1241" height="437" src="https://user-images.githubusercontent.com/42757351/129860363-fa36c5dd-c65d-4e2e-9eb7-55a2361c3e76.png">
</p>

## Sentiment Analysis Model

### 1. Question Definition

Identify the speaker's emotion based on their speech. 

### 2. Data Collection

* Large Movie Review Dataset - https://ai.stanford.edu/~amaas/data/sentiment/

* The structue of dataset

<p align="center">
  <img width="673" height="456" src="https://user-images.githubusercontent.com/42757351/129885881-5ffa79c7-f1a9-46ba-afee-fb4ee57ab113.png">
</p>

### 3. Data PreProcessing

Change <br /> to \n\n and remove whitespaces at the beginnig and end of the string. Then, find out their label 'pos' or 'neg'.

<p align="center">
  <img width="673" height="456" src="https://user-images.githubusercontent.com/42757351/129895971-0e9efbf6-5ef5-4933-ac47-0f7ae7cf2de1.png">
</p>


### 4. Model Building

After that, train the model.

<p align="center">
  <img width="1500" height="510" src="https://user-images.githubusercontent.com/42757351/129899336-acb91592-12eb-44bb-a09d-8dfc88af1268.png">
</p>

<p align="center">
  <img width="1241" height="437" src="https://user-images.githubusercontent.com/42757351/129899345-5ef62253-4ed5-48d7-934d-1f6c51f8d09d.png">
</p>


# 2. Server 

# 3. Web

# 4. Result

# 5. Timeline with Gantt chart 







