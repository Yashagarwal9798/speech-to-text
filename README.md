# Speech-to-Text from Scratch

## Overview

This project is a deep learning-based speech-to-text (ASR) system built from scratch using TensorFlow and Keras. It converts spoken language into written text using a custom-trained neural network.

## Features

- **End-to-End Speech Recognition**: Converts raw audio into text without pre-trained models.
- **Deep Learning Architecture**: Uses a Convolutional Recurrent Neural Network (CRNN) inspired by DeepSpeech2.
- **Custom Dataset Handling**: Loads and processes the LJSpeech dataset for training and evaluation.
- **CTC Loss for Sequence Learning**: Implements Connectionist Temporal Classification (CTC) loss to align predictions with text.
- **Word Error Rate (WER) Calculation**: Evaluates transcription accuracy.
- **Data Preprocessing**: Generates spectrograms from audio files.

## Dataset

The model is trained on the **LJSpeech-1.1 dataset**, which consists of paired audio and text transcriptions.

- **Audio Files**: Stored in `wavs/` directory.
- **Transcriptions**: Contained in `metadata.csv`.

## Installation

### Prerequisites

- Python 3.8+
- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib
- Jiwer (for WER calculation)

## Model Architecture

The model follows a DeepSpeech2-inspired architecture:

- **Convolutional Layers**: Extract features from spectrograms.
- **Bidirectional GRU Layers**: Capture sequential dependencies.
- **Dense Layers**: Convert features into character probabilities.
- **CTC Loss Function**: Enables alignment between audio and text.

## Evaluation Metrics

- **Word Error Rate (WER)**: Measures transcription accuracy.
- **Character Error Rate (CER)**: Additional evaluation for fine-grained accuracy.

## Future Improvements

- Implement beam search decoding for better transcription accuracy.
- Train on a larger dataset for improved generalization.
- Optimize model for real-time inference.


