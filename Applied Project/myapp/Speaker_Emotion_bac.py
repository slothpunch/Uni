#!/usr/bin/env python
# coding: utf-8
"""
* Title : Speaker Recognition
* Author : Fadi Badine
* Year : July 2020
* URL : https://keras.io/examples/audio/speaker_recognition_using_cnn/
* Dataset : https://www.kaggle.com/kongaevans/speaker-recognition-dataset
"""
import pdb

import os
import numpy as np

import tensorflow as tf
from tensorflow import keras

from pathlib import Path
from IPython.display import display, Audio
import librosa.display

import matplotlib.pyplot as plt

import speech_recognition as sr
# import text2emotion as te

import spacy # == install 2.3.5
import nltk # 3.6 or above required
from pprint import pprint

import keras
import random
from spacy.util import minibatch, compounding

# The main folder directory
DATASET_ROOT = os.path.join("speaker_recognition/16000_pcm_speeches")

# The folders in which we will put the audio samples and the noise samples
AUDIO_SUBFOLDER = "audio"
NOISE_SUBFOLDER = "noise"

DATASET_AUDIO_PATH = os.path.join(DATASET_ROOT, AUDIO_SUBFOLDER)
DATASET_NOISE_PATH = os.path.join(DATASET_ROOT, NOISE_SUBFOLDER)

# Seed to use when shuffling the dataset and the noise
SHUFFLE_SEED = 43

SAMPLING_RATE = 16000

SCALE = 0.5

BATCH_SIZE = 1 #128
EPOCHS = 20     #100

#
# Noise Preparation
#

# Get the list of all noise files
noise_paths = []
for subdir in os.listdir(DATASET_NOISE_PATH): # backnoise then go to other
    subdir_path = Path(DATASET_NOISE_PATH) / subdir
    if os.path.isdir(subdir_path): # If it is True,
        noise_paths += [
            os.path.join(subdir_path, filepath)
            for filepath in os.listdir(subdir_path) # 2 others and 4 background
            if filepath.endswith(".wav")
        ]

print(
    "Found {} files belonging to {} directories".format(
        len(noise_paths), len(os.listdir(DATASET_NOISE_PATH))
    )
    
)

#
# Resample all noise samples to 16000 Hz
#
# Split noise into chunks of 16000 each
def load_noise_sample(path):
    sample, sampling_rate = tf.audio.decode_wav(
        tf.io.read_file(path), desired_channels=1
    )

    if sampling_rate == SAMPLING_RATE:
        # Number of slices of 16000 each that can be generated from the noise sample
        slices = int(sample.shape[0] / SAMPLING_RATE/30)
        sample = tf.split(sample[: slices * SAMPLING_RATE], slices)
        return sample
    else:
        print(f"Incorrect smapling rate. {sampling_rate}")
        return None

noises = []
for path in noise_paths:
    sample = load_noise_sample(path)
    if sample:
        noises.extend(sample)
noises = tf.stack(noises)

print(
    "{} noise files were split into {} noise samples where each is {} sec. long".format(
        len(noise_paths), noises.shape[0], noises.shape[1] 
    )
)

#
# Dataset generation
#
def paths_and_labels_to_dataset(audio_paths, labels):
    """Constructs a dataset of audios and labels."""
    path_ds = tf.data.Dataset.from_tensor_slices(audio_paths)
    audio_ds = path_ds.map(lambda x: path_to_audio(x))
    label_ds = tf.data.Dataset.from_tensor_slices(labels)
    return tf.data.Dataset.zip((audio_ds, label_ds))


def path_to_audio(path):
    """Reads and decodes an audio file."""
    audio = tf.io.read_file(path)
    audio, _ = tf.audio.decode_wav(audio, 1, SAMPLING_RATE)
    return audio


def add_noise(audio, noises=None, scale=0.5):
    if noises is not None:
        # Create a random tensor of the same size as audio ranging from
        # 0 to the number of noise stream samples that we have.
        tf_rnd = tf.random.uniform(
            (tf.shape(audio)[0],), 0, noises.shape[0], dtype=tf.int32
        )
        noise = tf.gather(noises, tf_rnd, axis=0)

        # Get the amplitude proportion between the audio and the noise
        prop = tf.math.reduce_max(audio, axis=1) / tf.math.reduce_max(noise, axis=1)
        prop = tf.repeat(tf.expand_dims(prop, axis=1), tf.shape(audio)[1], axis=1)

        # Adding the rescaled noise to audio
        audio = audio + noise * prop * scale

    return audio


def audio_to_fft(audio):
    # Since tf.signal.fft applies FFT on the innermost dimension,
    # we need to squeeze the dimensions and then expand them again after FFT
    audio = tf.squeeze(audio, axis=-1)
    fft = tf.signal.fft(
        tf.cast(tf.complex(real=audio, imag=tf.zeros_like(audio)), tf.complex64)
    )
    fft = tf.expand_dims(fft, axis=-1)

    # Return the absolute value of the first half of the FFT
    # which represents the positive frequencies
    return tf.math.abs(fft[:, : (audio.shape[1] // 2), :])
    
# Get the list of audio file paths along with their corresponding labels

class_names = os.listdir(DATASET_AUDIO_PATH)
# print("Our class names: {}".format(class_names,))

audio_paths = []
labels = []
for label, name in enumerate(class_names):
    dir_path = Path(DATASET_AUDIO_PATH) / name
    speaker_sample_paths = [
        os.path.join(dir_path, filepath)
        for filepath in os.listdir(dir_path)
        if filepath.endswith(".wav")
    ]
    audio_paths += speaker_sample_paths
    labels += [label] * len(speaker_sample_paths)

print(
    "Found {} files belonging to {} classes.".format(len(audio_paths), len(class_names))
)

# Count the number of samples
num_samples = int(len(audio_paths))

print("Using {} files for test.".format(num_samples))
test_audio_paths = audio_paths
test_labels = labels

test_ds = paths_and_labels_to_dataset(test_audio_paths, test_labels)
test_ds = test_ds.shuffle(buffer_size=32 * 8, seed=SHUFFLE_SEED).batch(32)

test_ds = test_ds.map(
    lambda x, y: (audio_to_fft(x), y), num_parallel_calls=tf.data.experimental.AUTOTUNE
)
test_ds = test_ds.prefetch(tf.data.experimental.AUTOTUNE)


# Load the trained model
model = tf.keras.models.load_model("speaker_model.h5")

#
# Deomstration
#
SAMPLES_TO_DISPLAY = 1

test_ds = paths_and_labels_to_dataset(test_audio_paths, test_labels)
test_ds = test_ds.shuffle(buffer_size=BATCH_SIZE * 8, seed=SHUFFLE_SEED).batch(
    BATCH_SIZE
)
test_ds = test_ds.map(lambda x, y: (add_noise(x, noises, scale=SCALE), y))
pred_speaker = [] # store the predicted speaker

for audios, labels in test_ds.take(1):
    # Get the signal FFT
    ffts = audio_to_fft(audios)
    # Predict
    y_pred = model.predict(ffts)
    # Take random samples
    rnd = np.random.randint(0, BATCH_SIZE, SAMPLES_TO_DISPLAY)
    audios = audios.numpy()[rnd, :, :]#
    labels = labels.numpy()[rnd]
    y_pred = np.argmax(y_pred, axis=-1)[rnd]

    for index in range(SAMPLES_TO_DISPLAY):
        pred_speaker.append(class_names[y_pred[index]])
        # For every sample, print the true and predicted label
        # as well as run the voice with the noise
        print(
            "Speaker: {} - Predicted: {}".format(
                class_names[labels[index]],
                class_names[y_pred[index]],
            )
        )
#         display(Audio(audios[index, :, :].squeeze(), rate=SAMPLING_RATE))

# Shows FFT
def plot_magnitutde_spectrum(audio_paths, title, sr, f_ratio=1):
    
    signal, _ = librosa.load(audio_paths, sr=SAMPLING_RATE)
    
    ft = np.fft.fft(signal)
    magnitude_spectrum = np.abs(ft)

    # plot magnitude spectrum
    plt.figure(figsize=(18,5))
    
    frequency = np.linspace(0, sr, len(magnitude_spectrum))
    num_frequency_bins = int(len(frequency) * f_ratio)
    plt.plot(frequency[:num_frequency_bins], magnitude_spectrum[:num_frequency_bins])
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.title(title)
    
    plt.show()

# plot_magnitutde_spectrum(audio_paths[0], pred_speaker[0], SAMPLING_RATE, 0.1)

#
# Sentiment Analysis
#
r = sr.Recognizer()
speaker_audio = sr.AudioFile(audio_paths[0])
with speaker_audio as source:
    audio = r.record(source)
speaker_text = r.recognize_google(audio)

def test_model(input_data: str = speaker_text):
    # Load saved trained model
    loaded_model = spacy.load("model_artifacts")
    # Generate prediction
    parsed_text = loaded_model(input_data)
    # Determine prediction to return
    if parsed_text.cats["pos"] > parsed_text.cats["neg"]:
        prediction = "Positive"
        score = parsed_text.cats["pos"]
    else:
        prediction = "Negative"
        score = parsed_text.cats["neg"]
    print(
        f"Review text:\n{input_data}\n\nPredicted sentiment: {prediction},"
        f"\tScore: {round(score,5)}"
    )


# Print out the results from speaker recognition and sentiment analysis with the input audio file
print(f'Predicted speaker : {pred_speaker[0]}')
print('')
test_model()
display(Audio(audio_paths[0], rate=SAMPLING_RATE))

