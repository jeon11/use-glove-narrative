# About 
Using vector space models (Google's Universal Sentence Encoder & GloVe) on word/phrase/sentence level in narrative stories. We measure cosine similarity on words as demo, and plot similarity matrix. As this repo is primarily for demo purposes, it contains only 1 set of data from the narrative dataset.


## Getting Started
**More details in [Notebook](https://github.com/jeon11/use-glove-narrative/blob/master/use_glove_cosine_similarity.ipynb#Installation-&-Setup)**

1. Clone the repo (has sample dataset and GloVe vectors):
```
git clone https://github.com/jeon11/use-glove-narrative.git
```

2. It is recommended to use Anaconda. You can simply create a new environment for tensorflow
After installing Anaconda, create a new environment:
```
conda create -n py3 python=3.6.8
```
Then activate the environment.
```
conda activate py3
```

3. Using pip, install packages for pandas, numpy, seaborn, tensorflow, tensorflow-hub
(ie. pip install pckge-name)
```
pip install --upgrade tensorflow=1.15.0
pip install --upgrade tensorflow-hub=0.7.0
```

Once the steps are done, we should be able to run the codes locally. Refer to the notebook.

### Hard Requirements
python=3.6.x
tensorflow==1.15.0
tensorflow-hub==0.7.0
