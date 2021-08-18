# Code sources for the model
**Speaker Recognition** - https://keras.io/examples/audio/speaker_recognition_using_cnn/#data-preparation

**Speech Recognition** - Python Library (called SpeachRecognition)

**Sentiment Analysis** - https://realpython.com/sentiment-analysis-python/#using-machine-learning-classifiers-to-predict-sentiment

# 1. INTRODUCTION

This is the final report of the project: build a speaker recognition and mood analysis system. The purpose of the system is to find a speaker in a given dataset and performing text-independent mood analysis. We employed AWS services, JSON, Python (an open-source programming language), and other tools. Initially, we were planning to use a couple of AWS services for machine learning, but we decided to use Colab after all. 

### FFT
Fast Fourier Transform (FFT) is used for speaker recognition and the result of each speaker shows the difference in the frequency-domain representation. The accuracy in both speaker recognition and mood analysis is high 0.95% and 0.89% respectively. 

### Limitations
* Only the five speakers: Benjamin Netanyahu, Jens Stoltenberg, Julia Gillard, Margaret Thatcher, and Nelson Mandela
* Only 30-second .wav format audio files are available 

### Problem 
* The model for speaker recognition is trained with 1-second audio files and when it comes the different length of audio files, the model often fails to predict the correct speaker.
* The sound of the audio files converted from video files is not equal. The results of FFT are not equal even though the speaker is the same. This leads the model to the wrong prediction. 

### Programming Environment
Colab Pro

### Dataset
Kaggle - https://www.kaggle.com/kongaevans/speaker-recognition-dataset

Youtube












