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

### Problem 
* The model for speaker recognition is trained with 1-second audio files and when it comes the different length of audio files, the model often fails to predict the correct speaker.
* The sound of the audio files converted from video files is not equal. The results of FFT are not equal even though the speaker is the same. This leads the model to the wrong prediction. 

### Dataset
* Kaggle - https://www.kaggle.com/kongaevans/speaker-recognition-dataset

* Youtube

![image](https://user-images.githubusercontent.com/42757351/129858450-b334f058-f995-40ac-871a-f4a712a38d49.png)

* audiocheck 

![image](https://user-images.githubusercontent.com/42757351/129858463-00d3080a-9e1c-41a0-91e7-d5eb4267ed24.png)

# 1. Model

* 

* Number of audio files for each speaker 
 
![image](https://user-images.githubusercontent.com/42757351/129860909-4ab0773f-6248-46dd-a0cf-28fa57777c13.png)



* Model accuracy with graph

![image](https://user-images.githubusercontent.com/42757351/129860363-fa36c5dd-c65d-4e2e-9eb7-55a2361c3e76.png)



# 2. Server 

# 3. Web

# 4. Result

# 5. Timeline with Gantt chart 







