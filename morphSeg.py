import pandas as pd, numpy as np
import io
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
import matplotlib.pyplot as plt
import subprocess

from sklearn.metrics import accuracy_score
import tensorflow as tf

def unimorph(word, lang):

    #test = os.system("unimorph analyze " + word + " --lang " + lang)
    output = subprocess.check_output("unimorph analyze " + word + " --lang " + lang, shell=True)
    print(output.decode())
    if output.decode() == "":
        morph_cat = "_"
    else:
        morph_cat = output.decode().split('\t')[2][:-1]
    print(morph_cat)
    #assert False
    return morph_cat

# Transform Data
def transform_word(row):
  char_list = []
  segs = str(row[2])
  text = str(row[1])
  #unimorph_cat = unimorph(text, 'eng')
  morph_boundary = 0
  for indx, char in enumerate(text):
    if len(segs) >= 1 and char != segs[0]:
        if segs[0] == " ":
            if len(segs) >= 4 and segs[:4] == ' @@' + char:
                morph_boundary = 1
                segs = segs[4:]
            elif len(segs) >= 3 and segs[:3] == ' @@':
                morph_boundary = 0
        else:
            if len(segs) >= 5 and segs[1:5] == ' @@' + char:
                morph_boundary = 1
                segs = segs[5:]
            elif len(segs) >= 6 and segs[2:6] == ' @@' + char:
                morph_boundary = 1
                segs = segs[6:]
            else:
                morph_boundary = 0
                segs = segs[1:]
    else:
        morph_boundary = 0
        segs = segs[1:]
    char_list.append({'word_index': row[0], 'char_index': indx, 'rev_char_index': len(text) - indx - 1,
                      'word_len': len(text), 'char_text': char, #'unimorph_cat': unimorph_cat,
                      'label' : morph_boundary})
  return char_list

def output_predictions(word_index, chars, preds): #, morph_cats

    lines = []
    line_index = 0
    word_text = ""
    segments = ""
    for index, char in enumerate(chars):
        if word_index[index] == line_index:
            # on same line
            word_text += char
            if preds[index]:
                # new morph
                segments += " @@" + char
            else:
                segments += char
        else:
            # on new line
            # save last line
            lines.append(word_text + '\t' + segments) # + '\t' + morph_cats[line_index]
            # reinitialize for next line
            line_index += 1
            word_text = char
            segments = char
    lines.append(word_text + '\t' + segments) #  + '\t' + morph_cats[line_index]

    # write to file
    with open('eng.word.dev.preds.tsv', 'w') as f:
        for line in lines:
            f.write(line + '\n')

    return

def main():
    # load the dataset - change to just reading strings to speed it up
    train_files = ["data/eng.word.train.tsv"]
    test_file = "data/eng.word.dev.tsv"
    train_input = pd.DataFrame()
    for train_file in train_files:
        new_train_input = pd.read_csv(train_file, sep="\t",quoting=3, names=['Text', 'Segments', 'Morph_Cat'], dtype=str)
        train_input = pd.concat([train_input, new_train_input], axis=0)
    test_input = pd.read_csv(test_file ,sep="\t",quoting=3, names=['Text', 'Segments', 'Morph_Cat'], dtype=str)

    #print(train_input.info())
    #print(train_input.head())

    train_list = []
    for row in train_input.itertuples():
        #print(row)
        train_list += transform_word(row)
    train = pd.DataFrame(train_list)

    test_list = []
    for row in test_input.itertuples():
        test_list += transform_word(row)
    test = pd.DataFrame(test_list) 

    #print(test.to_string())

    #output_predictions(test['word_index'], test['char_text'], test['label'])

    #print(train.info())
    #print(train.head())

    #assert False

    train_hot = pd.get_dummies(train)
    train_hot.head()

    test_hot = pd.get_dummies(test)

    # Get missing columns in test
    missing_cols = set(train_hot.columns) - set(test_hot.columns)

    # Add a missing column in test set with default value equal to 0
    for c in missing_cols:
        test_hot[c] = 0

    # Get test with only columns from train and in same order
    test_hot = test_hot[train_hot.columns]
    print(test_hot.head())

    X_train = train_hot.drop(["label","word_index"],axis=1)
    y_train = train_hot["label"]
    X_test = test_hot.drop(["label", "word_index"],axis=1)
    y_test = test_hot["label"]

    print(X_train.head())
    print(y_test)

    # Standard scaling layer
    normalizer = tf.keras.layers.Normalization(axis=-1)
    normalizer.adapt(X_train)

    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)

    model = tf.keras.Sequential([
        normalizer,
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(50, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    # Model compilation
    model.compile(optimizer='adam',
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy'])

    # Basic TF2 keras API is nice and sklearn-like
    model.fit(X_train, y_train, epochs=10, batch_size=32, callbacks=[callback])

    model.evaluate(X_test,y_test, return_dict=True)

    preds = model.predict(X_test)
    preds = np.where(preds>0,1,0).ravel()
    
    output_predictions(test['word_index'], test['char_text'], preds)

    model.save("morph_seg_model")

if __name__ == "__main__":
    main()
