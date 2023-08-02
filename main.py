import tensorflow as tf
import numpy as np
import librosa
import sklearn
import glob
import sklearn.metrics
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.sequence import pad_sequences

# Define some constants
SAMPLE_RATE = 22050 # The sampling rate of the audio files in Hz
NUM_CLASSES = 28 # The number of classes you want to classify
NUM_MFCCS = 13 # The number of MFCCs to extract from each audio frame
FRAME_LENGTH = 2048 # The length of each audio frame in samples
HOP_LENGTH = 512 # The hop length between successive frames in samples
TRAIN_FOLDER = "noise_train" # The path to the training data folder
PATTERN = "*.wav" # The file extension of the data files
TEST_FOLDER = "noise_test" # The path to the test data folder
FIXED_LENGTH = 100 # Define a fixed length for the arrays


# Define a function to load and preprocess an audio file
def load_and_preprocess_audio(file_path):
  # Load the audio file as a numpy array
  audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)
  # Normalize the audio data
  audio = audio / np.max(np.abs(audio))
  # Extract the MFCCs from the audio signal
  mfccs = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=NUM_MFCCS,
                             n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)
  # Transpose the MFCCs to have the time dimension first
  mfccs = mfccs.T
  # Return the MFCCs as a numpy array
  return mfccs

# Define a function to get the label from the file name
def get_label_from_file_name(file_path):
  # Split the file path by the slash character
  file_path_parts = file_path.split("\\")
  # Get the last part of the file path, which is the file name
  file_name = file_path_parts[-1]
  # Split the file name by the underscore character
  file_name_parts = file_name.split("_")
  # # Get the first part of the file name, which is the label
  label = file_name_parts[0]
  # Return the label as a string
  return label

# Define a list of paths to your training audio files
train_files = glob.glob(TRAIN_FOLDER + "/" + PATTERN)

# Define an empty list to store the labels
train_labels = []

# Loop over the train_files list and get the label for each file
for file_path in train_files:
  # Get the label from the file name
  label = get_label_from_file_name(file_path)
  # Append the label to the train_labels list
  train_labels.append(label)

# Define an empty list to store the preprocessed training data
train_data = []

# Loop over the training files and preprocess them
for file_path in train_files:
  print("FILE: " + file_path)
  # Load and preprocess the audio file
  mfccs = load_and_preprocess_audio(file_path)
  # Append the MFCCs to the train_data list
  train_data.append(mfccs)

# Convert the train_data list to a numpy array
# train_data = np.array(train_data, dtype=object)
train_data = pad_sequences(train_data, maxlen=FIXED_LENGTH, padding='post', truncating='post')

# Do the same for the test audio files
test_files = glob.glob(TEST_FOLDER + "/" + PATTERN)
# Fit the label encoder with the test_labels list
# Define an empty list to store the labels
test_labels = []

# Loop over the test_files list and get the label for each file
for file_path in test_files:
  # Get the label from the file name
  label = get_label_from_file_name(file_path)
  # Append the label to the test_labels list
  test_labels.append(label)
  
# Create an instance of LabelEncoder
label_encoder = LabelEncoder()
# Gather all the labels into one variable
all_labels = train_labels + test_labels
# Encode all possible labels
label_encoder.fit(all_labels)
# Transform the train_labels list to numerical labels
train_labels = label_encoder.transform(train_labels)
# Transform the test_labels list to numerical labels
test_labels = label_encoder.transform(test_labels)

test_data = []
for file_path in test_files:
  print(file_path)
  mfccs = load_and_preprocess_audio(file_path)
  test_data.append(mfccs)
test_data = pad_sequences(test_data, maxlen=FIXED_LENGTH, padding='post', truncating='post')

# Define some hyperparameters
BATCH_SIZE = 32 # The number of samples per batch for training
EPOCHS = 200 # The number of epochs (iterations over the whole dataset) for training
LEARNING_RATE = 0.001 # The learning rate for the optimizer

# Define a function to create a CNN model
def create_cnn_model(input_shape, num_classes):
  # Create a sequential model
  model = tf.keras.models.Sequential()
  # Add a convolutional layer with 32 filters, kernel size of 3x3,
  # padding='same', and ReLU activation
  model.add(tf.keras.layers.Conv2D(32, (3, 3), padding='same',
                                   activation='relu', input_shape=input_shape))
  # Add a batch normalization layer
  model.add(tf.keras.layers.BatchNormalization())
  # Add a max pooling layer with pool size of 2x2
  model.add(tf.keras.layers.MaxPooling2D((2, 2)))
  # Add a dropout layer with dropout rate of 0.25
  model.add(tf.keras.layers.Dropout(0.25))
  model.add(tf.keras.layers.Conv2D(64, (3, 3), padding='same',
                                   activation='relu'))
  model.add(tf.keras.layers.BatchNormalization())
  model.add(tf.keras.layers.MaxPooling2D((2, 2)))
  model.add(tf.keras.layers.Dropout(0.25))
  # Flatten the output of the last convolutional layer
  # Add a GlobalMaxPooling2D layer to reduce the spatial dimensions of the output of the MaxPooling2D layer
  model.add(tf.keras.layers.GlobalMaxPooling2D())
  # Add a dense layer with 128 units and ReLU activation
  model.add(tf.keras.layers.Dense(128, activation='relu'))
  # Add a dropout layer with dropout rate of 0.5
  model.add(tf.keras.layers.Dropout(0.5))
  # Add a dense layer with num_classes units and softmax activation
  model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
  # Return the model
  return model

# Create the CNN model
model = create_cnn_model(input_shape=(None, NUM_MFCCS, 1), num_classes=NUM_CLASSES)

# Compile the model with categorical crossentropy loss,
# Adam optimizer, and accuracy metric
model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
              metrics=['accuracy'])

# Print the model summary
model.summary()

# Convert the labels to one-hot encoded vectors
train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=NUM_CLASSES)
test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=NUM_CLASSES)

# Reshape the data to have an extra dimension for the channels
train_data = train_data[..., np.newaxis]
test_data = test_data[..., np.newaxis]

# Train the model with the train_data and train_labels,
# using the test_data and test_labels for validation,
# and the batch size and epochs defined earlier
model.fit(train_data, train_labels,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          validation_data=(test_data, test_labels))

# Evaluate the model on the test data and print the test loss and accuracy
test_loss, test_acc = model.evaluate(test_data, test_labels)
print('Test loss:', test_loss)
print('Test accuracy:', test_acc)

# Predict the probabilities for each test sample
test_probs = model.predict(test_data)

# Get the predicted labels by taking the argmax of the probabilities
test_preds = np.argmax(test_probs, axis=1)

# Get the true labels by taking the argmax of the one-hot encoded vectors
test_true = np.argmax(test_labels, axis=1)

# Compute and print the confusion matrix using sklearn.metrics.confusion_matrix
conf_matrix = sklearn.metrics.confusion_matrix(test_true, test_preds)
print('Confusion matrix:\n', conf_matrix)

# Compute and print the precision, recall, and f1-score for each class using sklearn.metrics.classification_report
class_report = sklearn.metrics.classification_report(test_true, test_preds)
print('Classification report:\n', class_report)

# Predict the probabilities for each test sample
test_probs = model.predict(test_data)

# Get the predicted labels by taking the argmax of the probabilities
test_preds = np.argmax(test_probs, axis=1)

# Get the original labels by using the label_encoder.inverse_transform method
test_labels = label_encoder.inverse_transform(test_preds)

# Loop over the test_files list and print or display each file path and its predicted label
for i in range(len(test_files)):
  # Get the file path and the label at index i
  file_path = test_files[i]
  label = test_labels[i]
  # Print or display the file path and the label
  print(file_path, label)