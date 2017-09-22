
#Authors: Neele Witte, 4067845; Patricia Fischer, 3928367
#Honor Code:  We pledge that this program represents our own work.


import sys
from os import listdir
import numpy as np
import tensorflow as tf
from gensim.models.keyedvectors import KeyedVectors

from config import DefaultConfig
from model import Model, Phase
from numberer import Numberer
from viterbi_decoder import Viterbi_Decoder
from scorer import Scorer


# Global variable data, needed in order to only iterate once
data = []

"""
This method extracts a dictionary that contains all words in the vocabulary and it's suitable word index
from the word embeddings
"""
def word2index(model):
    counter= 0
    word2index = {}
    for key in model.wv.index2word:
        word2index[key] = counter
        counter+=1
    word2index['unknown']=counter
    word2index['noWord']=counter+1
    return word2index

"""
This method reads the input data. It returns a matrix that contains every word index and the appropriate label encoded
as integer.
For example:
'A house' would be encoded as
[(3, 15), (10, 15)] where 3 and 10 are word indices for 'A' and 'house' and will be mapped to their word vector later on and 15
is the label encoding for 'O' which is the suitable named entity tag
"""
def read_data(filename, embeddings, label_dict, word2index):
    with open(filename, "r") as f:
        sentence = []
        for line in f:
            if (line != "\n"):
                parts = line.split("\t")
                word = parts[1]
                tag = parts[5]
                wordindex, tagvec = convert_data(word, tag, embeddings, label_dict, word2index)
                sentence.append((wordindex, tagvec))
            else:
                data.append(sentence)
                sentence = []


'''
This mehtod extracts all named entity tags present in a corpus
'''
def get_labels(files):
    tags = []
    for file in files:
        with open(file, "r") as f:
            for line in f:
                if (line != "\n"):
                    parts = line.split("\t")
                    tag = parts[5]
                    tags.append(tag)
    return tags


'''
This mehtod encodes all named entity tags as a certain unique integer and returns two dictionaries
One contains the labels as keys mapped to their integer encoding, the other contains the integers
mapped to their labels
'''
def convert_label_to_number(labels):
    unique_labels = list(set(labels))
    label_to_number = dict()
    number_to_label = dict()
    l = Numberer()
    for label in unique_labels:
        n = l.number(label)
        label_to_number[label] = n
        number_to_label[n] = label
    return (label_to_number, number_to_label)

'''
This mehtod takes a word embedding matrix, l2 normalizes it and adds two word vectors.
One for an unknown word and one for a 'non-word'. Both new word vectors are randomly initialized
'''
def convert_word_Embeddings(wordEmbeddings):
    wordEmbeddings.wv.init_sims()
    newembeddings = np.zeros(
        shape=(
            len(wordEmbeddings.syn0norm)+2,
            len(wordEmbeddings.syn0norm[0])),
        dtype=np.float32)
    for lineindex in range(len(wordEmbeddings.syn0norm)):
        newembeddings[lineindex] = wordEmbeddings.syn0norm[lineindex]
    unknown = np.random.sample(100)
    noWord = np.random.sample(100)
    newembeddings[len(wordEmbeddings.syn0norm)] = unknown
    newembeddings[len(wordEmbeddings.syn0norm)+1] = noWord
    return newembeddings

'''
This method extracts all filenames given a certain directory
'''
def read_files(mypath):
    files = listdir(mypath)
    filenames = []
    for f in files:
        filenames.append(mypath+"/"+f)
    return filenames


'''
This method reads in pretrained word embeddings from a file in binary format
'''
def read_word_embeddings(f):
    word_vectors = KeyedVectors.load_word2vec_format(f, binary=True)
    return word_vectors


'''
This method takes a word and a tag as input and returns their integer representations as a Tuple.
Returns: (Wordindex, Labelindex)
'''
def convert_data(word, tag, embeds, tagdic, index2word):
    if (word in embeds.wv.vocab):
        wordindex = index2word[word]
    else:
        wordindex = index2word['unknown']

    return (wordindex, tagdic[tag])

'''
This method returns three matrices with the label the sentence lengths and the input data
'''
def generate_instances(data, n_labels, max_timesteps, batch_size=DefaultConfig.batch_size):
    n_batches = len(data) // batch_size

    labels = np.zeros(
        shape=(
            n_batches,
            batch_size,
            max_timesteps),
        dtype=np.int32)
    lengths = np.zeros(
        shape=(
            n_batches,
            batch_size),
        dtype=np.int32)
    sentences = np.full((n_batches, batch_size, max_timesteps), len(wordEmbeddings.syn0norm)+1, dtype=np.int32)


    for batch in range(n_batches):
        for idx in range(batch_size):
            sent = data[(batch*batch_size)+idx]
            s = [i[0] for i in sent]
            l = [i[1] for i in sent]

            # Sequence
            timesteps = min(max_timesteps, len(sent))

            # Sequence length (time steps)
            lengths[batch, idx] = timesteps

            # Labels
            labels[batch, idx, :timesteps] = l[:timesteps]

            # Words
            sentences[batch, idx,:timesteps] = s[:timesteps]

    return (sentences, lengths, labels)

'''
This methord starts a tensorflow session, feeds in all input data into the tensorflow graph and computes the loss and the the other
evaluation measures
'''
def train_model(config, train_batches, train_lens, train_labels,validation_batches, validation_lens, validation_labels, number_to_label, embeddings):

    with tf.Session() as sess:
        with tf.variable_scope("model", reuse=False):
            train_model = Model(
                config,
                train_batches,
                train_lens,
                train_labels,
                embeddings,
                phase=Phase.Train)

        with tf.variable_scope("model", reuse=True):
            validation_model = Model(
                config,
                validation_batches,
                validation_lens,
                validation_labels,
                embeddings,
                phase=Phase.Validation)

        sess.run(tf.global_variables_initializer())

        for epoch in range(config.n_epochs):
            train_loss = 0.0
            validation_loss = 0.0
            precision = 0.0
            recall = 0.0
            f1_score = 0.0

            # Train on all batches.
            for batch in range(train_batches.shape[0]):
                loss, _ = sess.run([train_model.loss,
                                    train_model.train_op], {
                                       train_model.x: train_batches[batch],
                                       train_model.lens: train_lens[batch],
                                       train_model.y: train_labels[batch]})
                train_loss += loss
                print("Trained on batch "+str(batch))

            print("Training done")
            decoder = Viterbi_Decoder()
            scorer = Scorer()
            # Validate on all batches.
            for batch in range(validation_batches.shape[0]):
                loss, logits, transition_params = sess.run([validation_model.loss,
                                                            validation_model.logits,
                                                            validation_model.transition_params], {
                                                               validation_model.x: validation_batches[batch],
                                                               validation_model.lens: validation_lens[batch],
                                                               validation_model.y: validation_labels[batch]})
                print(logits.shape)
                validation_loss += loss
                print("Decode batch "+str(batch))
                viterbi_sequences = decoder.decode(logits, transition_params, validation_lens[batch])
                print("Calculate scores for batch "+str(batch))
                prec, rec, f1 = scorer.scores(viterbi_sequences, number_to_label, validation_labels, batch)
                # Get prec, rec and f1 for current batch
                precision += prec
                recall += rec
                f1_score += f1

            train_loss /= train_batches.shape[0]
            validation_loss /= validation_batches.shape[0]
            precision /= validation_batches.shape[0]
            recall /= validation_batches.shape[0]
            f1_score /= validation_batches.shape[0]

            print(
                "epoch %d - train loss: %.2f, validation loss: %.2f,"
                "validation precision: %.2f, validation recall: %.2f, validation f1-score: %.2f" %
                (epoch, train_loss, validation_loss, precision * 100, recall * 100, f1_score * 100))


if __name__ == "__main__":
    '''
    A directory with the Named Entity Corpus and files in conll format must be provided as first argument.
    As second argument has to contain the path to a file containing the pretrained word embeddings, it has to be
    binary format.
    '''
    if len(sys.argv) != 3:
        sys.stderr.write("Usage: %s DATA WORDEMBEDDINGS\n" % sys.argv[0])
        sys.exit(1)
    if (str(sys.argv[1]).endswith('bin') == False):
        sys.stderr.write('WORD EMBEDDING FILE HAS TO BE BINARY FORMAT')

    '''
    Arguments are read from command line. All conll files from the directory are saved in a list.
    '''
    filenames = read_files(sys.argv[1])
    embedding_file = sys.argv[2]

    '''
    The Named Entity Tags are extracted from the corpus and saved to a dictionary. All labels are converted into a unique integer
    '''
    tags = get_labels(filenames)
    (label_to_number, number_to_label) = convert_label_to_number(tags)

    '''
    Pretrained Word Embeddings are saved to a model. They are l2-normalized. All words in the corpus are converted into the
    word index mapped to the embedding. One embedding is added for any unknown word. It is randomly initialized. Another
    embedding is added for a 'Non-Word' which is used when padding the sentences to the same length, i.e. adding some non words in
    order to have each sentence having the same length.
    '''
    wordEmbeddings = read_word_embeddings(embedding_file)
    wordTwoindex = word2index(wordEmbeddings)
    for f in filenames:
        read_data(f, wordEmbeddings, label_to_number, wordTwoindex)

    embeddings = convert_word_Embeddings(wordEmbeddings)
    print("Embeddings have been read")

    '''
    Data is splittet into training and test data. For each batch a matrix with the labels, the real sentence lengths and the
    input data with word indices are created.
    '''
    training = data[0:25000]
    test = data[25001:]

    (train_sentences, train_lengths, train_labels) = generate_instances(
        training,
        len(label_to_number.keys())+1,
        DefaultConfig.max_timesteps,
        batch_size=DefaultConfig.batch_size)
    print(train_sentences)
    (validation_sentences, validation_lengths, validation_labels) = generate_instances(
        test,
        len(label_to_number.keys())+1,
        DefaultConfig.max_timesteps,
        batch_size=DefaultConfig.batch_size)
    print("Data has been read")

    '''The mode is ready to be trained'''
    train_model(DefaultConfig, train_sentences, train_lengths, train_labels, validation_sentences, validation_lengths, validation_labels, number_to_label, embeddings)
