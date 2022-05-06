import pandas as pd

def main(words, prefixes, suffixes):
	char_features = [] 
	# feature set: if char is the start of new morpheme
		# is the string to the left of the split a prefix
		# is the string to the right of the split a suffix
		# does a substring to the left contain a prefix (substring must start from end of string)
		# does a substring to the left contain a suffix (substring must start from end of string)
		# does a substring to the right contain a prefix (substring must start from start of string)
		# does a substring to the right contain a suffix (substring must start from start of string)
	for word in words:
		for i, char in enumerate(word):
			feature_set = ""
			# is the string to the left of the split a prefix
			if word[:i] in prefixes:
				feature_set += "1"
			else:
				feature_set += "0"
			# is the string to the right of the split a suffix
			if word[i:] in suffixes:
				feature_set += "1"
			else:
				feature_set += "0"
			# does a substring to the left contain a prefix (substring must start from end of string)
			left_pre = "0"
			# does a substring to the left contain a suffix (substring must start from end of string)
			left_suf = "0"
			y = len(word[:i])
			for j in range(y):
				if word[j:i] in prefixes:
					left_pre = "1"
				if word[j:i] in suffixes:
					left_suf = "1"
			feature_set += left_pre
			feature_set += left_suf
			# does a substring to the right contain a prefix (substring must start from start of string)
			right_pre = "0"
			# does a substring to the right contain a prefix (substring must start from end of string)
			right_suf = "0"
			z = len(word[i:])
			for k in range(z):
				if word[i:(i+k)] in prefixes:
					right_pre = "1"
				if word[i:(i+k)] in suffixes:
					right_suf = "1"
			feature_set += right_pre
			feature_set += right_suf
			char_features.append(feature_set)

	return char_features

def get_affix_lists(lang):
	prefix_file = "affix_list/" + lang + "_prefix.txt"
	suffix_file = "affix_list/" + lang + "_suffix.txt"
	with open(prefix_file, "r") as prefix:
		prefixes = set(prefix.read().split())
	with open(suffix_file, "r") as suffix:
		suffixes = set(suffix.read().split())

	return prefixes, suffixes

def get_data(lang, data_type):
	data_file = "data/" + lang + ".word." + data_type + ".tsv"
	with open(data_file, "r", encoding="utf-8") as f:
		lines = f.read().split("\n")
	words = []
	for line in lines:
		if line == "":
			continue
		word, _, _ = line.split("\t")
		words.append(word)

	return words

def write_to_file(lang, data_type, char_features):
	outfile = "char_features/" + lang + ".word." + data_type + ".features.txt"
	with open(outfile, "w+") as f:
		for feat_set in char_features:
			f.write(feat_set + "\n")
	return


if __name__ == "__main__":
	#datafile = "data/eng.word.train.tsv"
	#outfile = "char_features/eng.word.train.features.txt"
	prefixes, suffixes = get_affix_lists("eng")
	words = get_data("eng", "dev")
	char_features = main(words, prefixes, suffixes)
	write_to_file("eng", "dev", char_features)
