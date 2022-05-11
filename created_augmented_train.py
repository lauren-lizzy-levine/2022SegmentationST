
def main(main_lang, augmenting_langs, cats, include_dev=False):
	# also need to make char feat file
	# Get data
	main_file = "data/" + main_lang + ".word.train.tsv"
	main_feature_file = "char_features/" + main_lang + ".word.train.features.txt"

	augmenting_files = []
	augmenting_feat_files = []
	for lang in augmenting_langs: 
		augmenting_files.append("data/" + lang + ".word.train.tsv")
		augmenting_feat_files.append("char_features/" + lang + ".word.train.features.txt")
		if include_dev:
			augmenting_files.append("data/" + lang + ".word.dev.tsv")
			augmenting_feat_files.append("char_features/" + lang + ".word.dev.features.txt")
	if include_dev:
		augmenting_files.append("data/" + main_lang + ".word.dev.tsv")
		augmenting_feat_files.append("char_features/" + main_lang + ".word.dev.features.txt")

	with open(main_file, "r") as f:
		main_train = f.readlines()
	with open(main_feature_file, "r") as f:
		main_feautres = f.readlines()

	augmenting_train = []
	augmenting_features = []

	for file, feat_file in zip(augmenting_files, augmenting_feat_files):
		with open(file, "r") as f:
			lines = f.readlines()
		with open(feat_file, "r") as f:
			feat_lines = f.readlines()

		for line, feat_line in zip(lines, feat_lines):
			if line == "":
				continue
			#print(line)
			split_line = line.split("\t")
			if len(split_line) == 3:
				morph_cat = split_line[2]
			else:
				morph_cat = "X"
			if morph_cat[:-1] in cats:
				augmenting_train.append(line)
				augmenting_features.append(feat_line)
	augmenting_train += main_train
	augmenting_features += main_feautres 
	outfile = main_lang + ".augmented.word.train.tsv"
	feat_outfile = "char_features/" + main_lang + ".augmented.word.train.features.txt"
	with open(outfile, "w") as f:
		for word in augmenting_train:
			f.write(word)
	with open(feat_outfile, "w") as f:
		for feats in augmenting_features:
			f.write(feats)
	return

if __name__ == "__main__":
	main_lang = "spa"
	augmenting_langs = ["fra", "ita"]
	cats = ["000", "001", "010" "011"]
	main(main_lang, augmenting_langs, cats)