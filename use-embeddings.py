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

# due to some depecrated methods from change in tf1.x to tf2.0, I used specific sets of Python and tf versions
# tested on python == 3.6.8 | tensorflow == 1.15.0 | tensorflow_hub == 0.7.0

# script variables
# module_url = "https://tfhub.dev/google/universal-sentence-encoder/2"  #@param ["https://tfhub.dev/google/universal-sentence-encoder/2", "https://tfhub.dev/google/universal-sentence-encoder-large/3"]
baseDir = 'narrative_use'
embedding_size = 512

pwd = os.getcwd()
# define absolute directories
while os.path.basename(pwd) != baseDir:
    os.chdir('..')
    pwd = os.getcwd()
baseDir = pwd
dataDir = baseDir + '/stories'

# recursively find all csv files
all_csvs = [y for x in os.walk(dataDir) for y in glob(os.path.join(x[0], '*.csv'))]
all_csvs.sort()
# assert len(all_csvs) == 6

for csv in all_csvs:
    textfile = pd.read_csv(csv)
    title = os.path.basename(csv)[:-4]
    vector_df_columns = ['paragraph', 'index', 'text', 'size']
    for i in range(1, embedding_size + 1):
        vector_df_columns.append('dim' + str(i))
    vector_df = pd.DataFrame(columns=vector_df_columns)

    # Import the Universal Sentence Encoder's TF Hub module
    # for lite
    # embed = hub.Module(module_url)  # hub.load(module_url) for tf==2.0.0
    # for heavy
    embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder-large/3")
    # textfile.iloc[0]
    messages = []
    for t in range(0, len(textfile)):
        messages.append(textfile.iloc[t]['text'])
    # messages
    # Compute a representation for each message, showing various lengths supported.
    # word = "Elephant"
    # sentence1 = "Larry had a dream that Walters he could fly"
    # sentence2 = "Larry Walters had a dream that he could fly"
    # sentence2 = "The problem was that Larry's vision was poor"
    # sentence3 = "and he was thus ill-suited to become a pilot."
    # sentence4 = "Most people, faced with similar limitations, might let go of such a dream."
    # sentence5 = "Larry Walters was not most people."
    # messages = [sentence1, sentence2]
    # paragraph = (
        # "Universal Sentence Encoder embeddings also support short paragraphs. "
        # "There is no hard limit on how long the paragraph is. Roughly, the longer ")


    # Reduce logging output.
    logging.set_verbosity(logging.ERROR)

    with tf.compat.v1.Session() as session:
        session.run([tf.compat.v1.global_variables_initializer(), tf.compat.v1.tables_initializer()])
        message_embeddings = session.run(embed(messages))


    #-----------------------------------------------------
    # make sure all units are there
    assert len(message_embeddings) == len(textfile) == len(messages)

    print(type(message_embeddings))
    for e in range(0, len(message_embeddings)):
        vector_df.at[e, 'paragraph'] = textfile.iloc[e]['paragraph']
        vector_df.at[e, 'index'] = textfile.iloc[e]['index']
        vector_df.at[e, 'text'] = messages[e]
        vector_df.at[e, 'size'] = len(message_embeddings[e])
        for dim in range(0, len(message_embeddings[e])):
            vector_df.at[e, 'dim'+str(dim+1)] = message_embeddings[e][dim]

    vector_df.reindex(columns=vector_df_columns)
    vector_df.to_csv(title + '_vectors.csv',index=False)


# tess = pd.DataFrame(message_embeddings[0])
    # for i, message_embedding in enumerate(np.array(message_embeddings).tolist()):
    #     print("Message: {}".format(messages[i]))
    #     print("Embedding size: {}".format(len(message_embedding)))
    #     message_embedding_snippet = ", ".join((str(x) for x in message_embedding[:3]))
    #     print("Embedding: [{}, ...]\n".format(message_embedding_snippet))

# with tf.compat.v1.Session() as session:
#     # session.run([tf.initialize_all_variables().run(), tf.tables_initializer()])
#     # session.run([tf.Variable(0)])
#   # session.run([tf.global_variables_initializer(), tf.tables_initializer()])
#     message_embeddings = session.run(embed(messages))
# # tf.initialize_all_variables()
# # tf.global_variables_initializer()
#   for i, message_embedding in enumerate(np.array(message_embeddings).tolist()):
#     print("Message: {}".format(messages[i]))
#     print("Embedding size: {}".format(len(message_embedding)))
#     message_embedding_snippet = ", ".join(
#         (str(x) for x in message_embedding[:3]))
#     print("Embedding: [{}, ...]\n".format(message_embedding_snippet))
#


#-----------------------------------------
#-----------------------------------------
# def plot_similarity(labels, features, rotation):
#     corr = np.inner(features, features)
#     sns.set(font_scale=1.2)
#     g = sns.heatmap(corr, xticklabels=labels, yticklabels=labels, vmin=0, vmax=1, cmap="YlOrRd")
#     g.set_xticklabels(labels, rotation=rotation)
#     g.set_title("Semantic Textual Similarity")
#
#
# def run_and_plot(session_, input_tensor_, messages_, encoding_tensor):
#     message_embeddings_ = session_.run(encoding_tensor, feed_dict={input_tensor_: messages_})
#     plot_similarity(messages_, message_embeddings_, 90)
#
#
# messages = [
#     # Smartphones
#     "I like my phone",
#     "My phone is not good.",
#     "Your cellphone looks great.",
#
#     # Weather
#     "Will it snow tomorrow?",
#     "Recently a lot of hurricanes have hit the US",
#     "Global warming is real",
#
#     # Food and health
#     "An apple a day, keeps the doctors away",
#     "Eating strawberries is healthy",
#     "Is paleo better than keto?",
#
#     # Asking about age
#     "How old are you?",
#     "what is your age?",
# ]
#
# similarity_input_placeholder = tf.compat.v1.placeholder(tf.string, shape=(None))
# similarity_message_encodings = embed(similarity_input_placeholder)
# with tf.compat.v1.Session() as session:
#     session.run(tf.compat.v1.global_variables_initializer())
#     session.run(tf.compat.v1.tables_initializer())
#     run_and_plot(session, similarity_input_placeholder, messages, similarity_message_encodings)
