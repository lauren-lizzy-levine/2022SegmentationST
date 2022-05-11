import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import Concatenate
import random

# configuration
batch_size = 64  # Batch size for training.
epochs = 30  # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space.
num_samples = 10000  # Number of samples to train on.
# Path to the data txt file on disk.
data_path = "spa.augmented.word.train.tsv"
feature_path = "char_features/fra.word.train.features.txt"

# Get data features
with open(feature_path, "r", encoding="utf-8") as f:
	char_features = f.read().split("\n")

# Prepare the data
# Vectorize the data.
input_texts = []
target_texts = []
morph_cats = []
input_characters = set()
target_characters = set()
with open(data_path, "r", encoding="utf-8") as f:
	lines = f.read().split("\n")
for line in lines: #[: min(num_samples, len(lines) - 1)]: # all data for the real deal
	if line == "":
		continue
	if len(line.split("\t")) == 2:
		input_text, target_text = line.split("\t")
		morph_cat = "NA"
	else:
		input_text, target_text, morph_cat = line.split("\t")
	# We use "tab" as the "start sequence" character
	# for the targets, and "\n" as "end sequence" character.
	target_text = "\t" + target_text + "\n"
	input_texts.append(input_text)
	target_texts.append(target_text)
	morph_cats.append(morph_cat)
	for char in input_text:
		if char not in input_characters:
			input_characters.add(char)
	for char in target_text:
		if char not in target_characters:
			target_characters.add(char)

input_characters = sorted(list(input_characters))
target_characters = sorted(list(target_characters))
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)
max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])

print("Number of samples:", len(input_texts))
print("Number of unique input tokens:", num_encoder_tokens)
print("Number of unique output tokens:", num_decoder_tokens)
print("Max sequence length for inputs:", max_encoder_seq_length)
print("Max sequence length for outputs:", max_decoder_seq_length)

input_token_index = dict([(char, i) for i, char in enumerate(input_characters)])
target_token_index = dict([(char, i) for i, char in enumerate(target_characters)])

encoder_input_data = np.zeros(
	(len(input_texts), max_encoder_seq_length, num_encoder_tokens), dtype="float32" # + 6
)
decoder_input_data = np.zeros(
	(len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype="float32"
)
decoder_target_data = np.zeros(
	(len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype="float32"
)

char_index = 0
# shuffle train order
joint = list(zip(input_texts, target_texts))
random.shuffle(joint)
input_texts, target_texts = zip(*joint)

for i, (input_text, target_text) in enumerate(zip(list(input_texts), list(target_texts))):
	for t, char in enumerate(input_text):
		encoder_input_data[i, t, input_token_index[char]] = 1.0
		#for k, digit in enumerate(char_features[char_index]):
		#	encoder_input_data[i, t, num_encoder_tokens + k] = float(digit)
		char_index += 1
	encoder_input_data[i, t + 1 :, input_token_index[" "]] = 1.0
	for t, char in enumerate(target_text):
		# decoder_target_data is ahead of decoder_input_data by one timestep
		decoder_input_data[i, t, target_token_index[char]] = 1.0
		if t > 0:
			# decoder_target_data will be ahead by one timestep
			# and will not include the start character.
			decoder_target_data[i, t - 1, target_token_index[char]] = 1.0
	decoder_input_data[i, t + 1 :, target_token_index[" "]] = 1.0
	decoder_target_data[i, t:, target_token_index[" "]] = 1.0


# Build the Model
# Define an input sequence and process it.
encoder_inputs = keras.Input(shape=(None, num_encoder_tokens)) # + 6
encoder = keras.layers.GRU(latent_dim, return_state=True)
encoder_outputs, state_h = encoder(encoder_inputs) #state_c
#state_h = Concatenate()([f_state_h, b_state_h])

# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = keras.Input(shape=(None, num_decoder_tokens))

# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
decoder_gru = keras.layers.GRU(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _ = decoder_gru(decoder_inputs, initial_state=state_h) # _
decoder_dense = keras.layers.Dense(num_decoder_tokens, activation="softmax")
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Early Stoping
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

# Training
model.compile(
	optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"]
)
model.fit(
	[encoder_input_data, decoder_input_data],
	decoder_target_data,
	batch_size=batch_size,
	epochs=epochs,
	validation_split=0.2,
	callbacks=[callback]
)
# Save model
model.save("aug_spanish_s2s")
