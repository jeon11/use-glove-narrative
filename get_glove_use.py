from absl import logging
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from glob import glob
import re
import seaborn as sns
from numpy import dot
from numpy.linalg import norm
from operator import add

# due to some depecrated methods from change in tf1.x to tf2.0, I used specific sets of Python and tf versions
# tested on python == 3.6.8 | tensorflow == 1.15.0 | tensorflow_hub == 0.7.0

# script variables
module_url = "https://tfhub.dev/google/universal-sentence-encoder/2"  #@param ["https://tfhub.dev/google/universal-sentence-encoder/2", "https://tfhub.dev/google/universal-sentence-encoder-large/3"]
baseDir = 'use-glove-narrative'
# Import the Universal Sentence Encoder's TF Hub module
# for lite
embed = hub.Module(module_url)  # hub.load(module_url) for tf==2.0.0
# for hevvy
# embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder-large/3")
# embedding_size = 512
pwd = os.getcwd()
# define absolute directories
while os.path.basename(pwd) != baseDir:
    os.chdir('..')
    pwd = os.getcwd()
baseDir = pwd
dataDir = baseDir + '/data'
gloveDir = baseDir + '/glove'

os.chdir(gloveDir)
gloves = os.listdir(gloveDir)
gloves.sort()
gloves

# GloVe idea unit vectors files
iu_gloves = []
for file in gloves:
    if 'idea' in file:
        iu_gloves.append(file)
iu_gloves.sort()

os.chdir(dataDir)
stories =os.listdir(dataDir)
stories = [x for x in stories if x.endswith('.csv')]
stories.sort()


def glove_vec(item1, item2):
    """
    get vectors for given two words and calculate cosine similarity

    Parameters
    ----------
    item1 : str
        string in glove word pool vector to compare
    item2 : str
        string in glove word pool vector to compare

    Returns
    -------
    item1_vector : array
        item1 GloVe vector
    item2_vector : array
        item2 GloVe vector
    cos_sim : float
        cosine similarity of item1 and item2 vectors

    """
    os.chdir(gloveDir)
    # change gloves[0] later
    lines = open(gloves[0]).read().split('\n')
    for i in range(0, len(lines)):
        if item1 in lines[i]:
            item1_vector = lines[i].split(' ')[1:]  # remove the text label
        elif item2 in lines[i]:
            item2_vector = lines[i].split(' ')[1:]  # remove the text label

    # convert str to floats
    item1_vector = [float(i) for i in item1_vector]
    item2_vector = [float(i) for i in item2_vector]

    # calculate cos sim
    cos_sim = dot(item1_vector, item2_vector)/(norm(item1_vector) * norm(item2_vector))

    return item1_vector, item2_vector, cos_sim


def use_vec(item1, item2):
    """
    get USE vectors and cosine similairty of the two items

    Parameters
    ----------
    item1 : str, list
        any word to compare, put in string for more than one word
    item2 : str, list
        any word to compare, put in string for more than one word

    Returns
    -------
    item1_vector : array
        item1 USE vector
    item2_vector : array
        item2 USE vector
    cos_sim : float
        cosine similarity of item1 and item2 vectors

    """
    # Import the Universal Sentence Encoder's TF Hub module
    embed = hub.Module(module_url)  # hub.load(module_url) for tf==2.0.0

    messages = [item1, item2]
    with tf.compat.v1.Session() as session:
        session.run([tf.compat.v1.global_variables_initializer(), tf.compat.v1.tables_initializer()])
        message_embeddings = session.run(embed(messages))

    item1_vector = message_embeddings[0]
    item2_vector = message_embeddings[1]

    cos_sim = dot(item1_vector, item2_vector)/(norm(item1_vector) * norm(item2_vector))

    return item1_vector, item2_vector, cos_sim


def compare_word_vec(item1, item2):
    """
    Given two strings of words, prints the GloVe-based cosine similarity and USE-based cosine similarity

    Parameters
    ----------
    item1 : str
        string of word to compare
    item2 : str
        string of word to compare

    Returns
    -------
    prints cosine similarities of GloVe-based vectors and USE-based vectors

    """
    g1, g2, gcos = glove_vec(item1, item2)
    u1, u2, ucos = use_vec(item1, item2)
    print('use cos: ' + str(ucos))
    print('glove cos: ' + str(gcos))


def iu_vec(iu1, iu2, story=1, embed=embed):
    """
    comparing phrase/sentence level/idea unit to idea unit within a single narrative story

    Parameters
    ----------
    iu1 : int
        idea unit in integer to compare
    iu2 : int
        idea unit in integer to compare
    story : int
        story integer (specific to narrative dataset)
    embed : str
        link to which USE model to employ

    Returns
    -------
    prints cosine similarities of GloVe-based vectors and USE-based vectors

    """
    story = str(story)
    # load GloVe vector files first
    os.chdir(gloveDir)
    for file in iu_gloves:
        if story in file:
            lines = open(file).read().split('\n')
            break

    g1 = lines[iu1].split(' ')
    for i in range(0, len(g1)):  # idea unit vectors have misc '' at the end of each vector so we remove them
        if g1[i] == '':
            del g1[i]
    g1 = [float(i) for i in g1]

    g2 = lines[iu2].split(' ')
    for i in range(0, len(g2)):  # idea unit vectors have misc '' at the end of each vector so we remove them
        if g2[i] == '':
            del g2[i]
    g2 = [float(i) for i in g2]
    # except ValueError:
    #     print("found '' empty string.passing")
    glove_cos_sim = dot(g1, g2)/(norm(g1) * norm(g2))

    # load USE idea units csv
    os.chdir(dataDir)
    for file in stories:
        if story in file:
            iu_use = pd.read_csv(file)
            break

    u1_text = iu_use.iloc[iu1]['text']
    u2_text = iu_use.iloc[iu2]['text']
    messages = [u1_text, u2_text]
    with tf.compat.v1.Session() as session:
        session.run([tf.compat.v1.global_variables_initializer(), tf.compat.v1.tables_initializer()])
        message_embeddings = session.run(embed(messages))

    item1_vector = message_embeddings[0]
    item2_vector = message_embeddings[1]

    use_cos_sim = dot(item1_vector, item2_vector)/(norm(item1_vector) * norm(item2_vector))

    print('glove sim: ' + str(glove_cos_sim))
    print('USE sim: ' + str(use_cos_sim))


def glove_phrase_vec(wordlist, story=1):
    """Short summary.

    Parameters
    ----------
    wordlist : list
        Description of parameter `wordlist`.
    story : type
        Description of parameter `story`.

    Returns
    -------
    type
        Description of returned object.

    """
    story = str(story)
    # load GloVe vector files first
    os.chdir(gloveDir)
    for file in gloves:
        if story in file and 'glove' in file:
            lines = open(file).read().split('\n')
            break

    # we first create a list that shows how many word vectors are extracted
    glove_vectors = []
    for i in range(0, len(wordlist)):
        glove_vectors.append('g' + str(i+1))

    # let's find the corresponding vectors for given words
    for i in range(0, len(wordlist)):
        current_word = wordlist[i]
        for j in range(0, len(lines)):
            temp = ""
            if current_word in lines[j]:
                temp = 'g' + str(i+1) + '= lines[j].split(' ')[1:]'
                exec(temp)
    for i in range(0, len(glove_vectors)):
        temp = glove_vectors[i] + ' = np.array(eval(glove_vectors[i]))'
        exec(temp)

    for i in range(0, len(glove_vectors)):
        temp = ""
        temp_vec = glove_vectors[i]
        temp = temp_vec + " = [float(i) for i in eval(temp_vec)]"
        exec(temp)

    # now let's add all word vectors and divide by count of words
    sum = np.zeros(300)
    for i in range(0, len(glove_vectors)):
        try:
            sum += eval(glove_vectors[i])
        except IndexError:
            pass
    sum = sum/len(glove_vectors)
    return sum


def phrase_vec(wordlist1, wordlist2, story=1):
    """
    Comparing phrase level vectors in GloVe vs USE

    Parameters
    ----------
    wordlist1 : list
        list of words in string
    wordlist2 : list
        list of words in string
    story : int
        integer indicating which narrative story (specific to narrative dataset)

    Returns
    -------
    type
        prints cosine similarities of GloVe-based vectors and USE-based vectors

    """
    story = str(story)
    # get glove vectors for each phrase
    g1 = glove_phrase_vec(wordlist1,story)
    g2 = glove_phrase_vec(wordlist2,story)

    glove_cos_sim = dot(g1, g2)/(norm(g1) * norm(g2))

    # get USE vectors and cos sim
    flat_list1= ' '.join(wordlist1)
    flat_list2= ' '.join(wordlist2)
    messages = [flat_list1, flat_list2]

    with tf.compat.v1.Session() as session:
        session.run([tf.compat.v1.global_variables_initializer(), tf.compat.v1.tables_initializer()])
        message_embeddings = session.run(embed(messages))

    u1 = message_embeddings[0]
    u2 = message_embeddings[1]

    use_cos_sim = dot(u1, u2)/(norm(u1) * norm(u2))

    print('glove sim: ' + str(glove_cos_sim))
    print('USE sim: ' + str(use_cos_sim))


# plotting
def plot_similarity(labels, features, rotation):
    corr = np.inner(features, features)
    sns.set(font_scale=1.2)
    g = sns.heatmap(corr, xticklabels=labels, yticklabels=labels,
                    vmin=0, vmax=1, cmap="winter_r")  # YlOrRd #Blues
    g.set_xticklabels(labels, rotation=rotation)
    g.set_title("Semantic Textual Similarity")
    plt.show()


def run_and_plot(session_, input_tensor_, messages_, encoding_tensor):
    message_embeddings_ = session_.run(encoding_tensor, feed_dict={input_tensor_: messages_})
    plot_similarity(messages_, message_embeddings_, 90)


def plot_sim_matrix(messages):
    similarity_input_placeholder = tf.compat.v1.placeholder(tf.string, shape=(None))
    similarity_message_encodings = embed(similarity_input_placeholder)
    with tf.compat.v1.Session() as session:
        session.run(tf.compat.v1.global_variables_initializer())
        session.run(tf.compat.v1.tables_initializer())
        run_and_plot(session, similarity_input_placeholder, messages,
                   similarity_message_encodings)


##------------STS Auth
# def load_sts_dataset(filename):
#   # Loads a subset of the STS dataset into a DataFrame. In particular both
#   # sentences and their human rated similarity score.
#   sent_pairs = []
#   with tf.io.gfile.GFile(filename, "r") as f:
#     for line in f:
#       ts = line.strip().split("\t")
#       # (sent_1, sent_2, similarity_score)
#       sent_pairs.append((ts[5], ts[6], float(ts[4])))
#   return pandas.DataFrame(sent_pairs, columns=["sent_1", "sent_2", "sim"])
#
#
# def download_and_load_sts_data():
#   sts_dataset = tf.keras.utils.get_file(
#       fname="Stsbenchmark.tar.gz",
#       origin="http://ixa2.si.ehu.es/stswiki/images/4/48/Stsbenchmark.tar.gz",
#       extract=True)
#
#   sts_dev = load_sts_dataset(
#       os.path.join(os.path.dirname(sts_dataset), "stsbenchmark", "sts-dev.csv"))
#   sts_test = load_sts_dataset(
#       os.path.join(
#           os.path.dirname(sts_dataset), "stsbenchmark", "sts-test.csv"))
#
#   return sts_dev, sts_test
#
#
# sts_dev, sts_test = download_and_load_sts_data()
#
# sts_input1 = tf.compat.v1.placeholder(tf.string, shape=(None))
# sts_input2 = tf.compat.v1.placeholder(tf.string, shape=(None))
#
# # For evaluation we use exactly normalized rather than
# # approximately normalized.
# sts_encode1 = tf.nn.l2_normalize(embed(sts_input1), axis=1)
# sts_encode2 = tf.nn.l2_normalize(embed(sts_input2), axis=1)
# cosine_similarities = tf.reduce_sum(tf.multiply(sts_encode1, sts_encode2), axis=1)
# clip_cosine_similarities = tf.clip_by_value(cosine_similarities, -1.0, 1.0)
# sim_scores = 1.0 - tf.acos(clip_cosine_similarities)
#
# sts_data = sts_dev #@param ["sts_dev", "sts_test"] {type:"raw"}
#
# text_a = sts_data['sent_1'].tolist()
# text_b = sts_data['sent_2'].tolist()
# dev_scores = sts_data['sim'].tolist()
#
#
# def run_sts_benchmark(session):
#     """Returns the similarity scores"""
#     emba, embb, scores = session.run([sts_encode1, sts_encode2, sim_scores],
#                             feed_dict={sts_input1: text_a,sts_input2: text_b})
#     return scores
#
#
# with tf.Session() as session:
#     session.run(tf.compat.v1.global_variables_initializer())
#     session.run(tf.compat.v1.tables_initializer())
#     scores = run_sts_benchmark(session)
#
# pearson_correlation = scipy.stats.pearsonr(scores, dev_scores)
# print('Pearson correlation coefficient = {0}\np-value = {1}'.format(
#     pearson_correlation[0], pearson_correlation[1]))
