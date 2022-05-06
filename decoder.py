import numpy as np
import tensorflow as tf
from tensorflow import keras

latent_dim = 256 
num_samples = 10000
data_path = "data/fra.word.train.tsv"
feature_path = "char_features/fra.word.train.features.txt"

# Get data features
with open(feature_path, "r", encoding="utf-8") as f:
	char_features = f.read().split("\n")

# Prepare the data
# Vectorize the data.
input_texts = []
target_texts = []
input_characters = set()
target_characters = set()
with open(data_path, "r", encoding="utf-8") as f:
	lines = f.read().split("\n")
for line in lines[: min(num_samples, len(lines) - 1)]: # i guess we're training on very few samples rn
	input_text, target_text, morph_cat = line.split("\t")
	# We use "tab" as the "start sequence" character
	# for the targets, and "\n" as "end sequence" character.
	target_text = "\t" + target_text + "\n"
	input_texts.append(input_text)
	target_texts.append(target_text)
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

input_token_index = dict([(char, i) for i, char in enumerate(input_characters)])
target_token_index = dict([(char, i) for i, char in enumerate(target_characters)])


# Get novel input
data_path = "data/fra.word.dev.tsv"

# Prepare the data
# Vectorize the data.
input_texts = []
morph_cats = []
input_characters = set()
with open(data_path, "r", encoding="utf-8") as f:
	lines = f.read().split("\n")
for line in lines: # we need all the predictions!
	#print(line)
	if line == "":
		continue
	input_text, target_text, morph_cat = line.split("\t")
	morph_cats.append(morph_cat)
	input_texts.append(input_text)


novel_encoder_input_data = np.zeros(
	(len(input_texts), max_encoder_seq_length, num_encoder_tokens), dtype="float32" # + 6
)

char_index = 0
for i, input_text in enumerate(input_texts):
	#print(input_text)
	for t, char in enumerate(input_text):
		if char not in input_token_index: # lame unk guard
			char = 's'
		if t < max_encoder_seq_length: # lame truncation guard
			novel_encoder_input_data[i, t, input_token_index[char]] = 1.0
			#for k, digit in enumerate(char_features[char_index]):
			#	novel_encoder_input_data[i, t, num_encoder_tokens + k] = float(digit)
		char_index += 1
	novel_encoder_input_data[i, t + 1 :, input_token_index[" "]] = 1.0


# Define sampling models
# Restore the model and construct the encoder and decoder.
model = keras.models.load_model("s2s")

encoder_inputs = model.input[0]  # input_1
encoder_outputs, state_h_enc = model.layers[2].output  # gru_1 state_c_enc 
#state_h = Concatenate()([f_state_h, b_state_h])
encoder_states = [state_h_enc]
encoder_model = keras.Model(encoder_inputs, encoder_states)

decoder_inputs_t = model.input[1]  # input_2
decoder_inputs = tf.identity(decoder_inputs_t)
#decoder_inputs = model.input[1]  # input_2
decoder_state_input_h = keras.Input(shape=(latent_dim,))
#decoder_b_state_input_h = keras.Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h]
decoder_gru = model.layers[3]
decoder_outputs, state_h_dec = decoder_gru( # state_c_dec 
	decoder_inputs, initial_state=decoder_states_inputs
)
#state_h_dec = Concatenate()([f_state_h_dec, b_state_h_dec])
decoder_states = [state_h_dec]
decoder_dense = model.layers[4]
decoder_outputs = decoder_dense(decoder_outputs)
#decoder_model = keras.Model(inputs=[decoder_inputs].append(decoder_states_inputs), outputs=[decoder_outputs].append(decoder_states))
decoder_model = keras.Model(
	[decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states
)

# Reverse-lookup token index to decode sequences back to
# something readable.
reverse_input_char_index = dict((i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict((i, char) for char, i in target_token_index.items())


def decode_sequence(input_seq):
	# Encode the input as state vectors.
	states_value = encoder_model.predict(input_seq)

	# Generate empty target sequence of length 1.
	target_seq = np.zeros((1, 1, num_decoder_tokens))
	# Populate the first character of target sequence with the start character.
	target_seq[0, 0, target_token_index["\t"]] = 1.0

	# Sampling loop for a batch of sequences
	# (to simplify, here we assume a batch of size 1).
	stop_condition = False
	decoded_sentence = ""
	while not stop_condition:
		output_tokens, h = decoder_model.predict([target_seq] + [states_value]) #c

		# Sample a token
		sampled_token_index = np.argmax(output_tokens[0, -1, :])
		sampled_char = reverse_target_char_index[sampled_token_index]
		decoded_sentence += sampled_char

		# Exit condition: either hit max length
		# or find stop character.
		if sampled_char == "\n" or len(decoded_sentence) > max_decoder_seq_length:
			stop_condition = True

		# Update the target sequence (of length 1).
		target_seq = np.zeros((1, 1, num_decoder_tokens))
		target_seq[0, 0, sampled_token_index] = 1.0

		# Update states
		states_value = [h] # c
	return decoded_sentence


output = ""
outfile = "fra.word.dev.preds.seq2seq.tsv"

for seq_index in range(500): #range(len(novel_encoder_input_data) - 1):
	# Take one sequence (part of the training set)
	# for trying out decoding.
	input_seq = novel_encoder_input_data[seq_index : seq_index + 1]
	decoded_sentence = decode_sequence(input_seq)
	output += input_texts[seq_index] + "\t" + decoded_sentence
	print(seq_index)
with open(outfile, "w") as f:
		f.write(output)