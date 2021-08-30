
# DELFT


This repository contains the code for the thesis Enhancing Question Answering with a Free-text Knowledge Graph. This work was aimed to further imrpove the ability of DELFT introduced by [Complex Question Answering with a Free-text Knowledge Graph](https://arxiv.org/abs/2103.12876). The original repository can be found [here](https://github.com/henryzhao5852/DELFT).
# Overview

<br><br>
<div align="center">
<img src="DELFT.png" width="400" />
</div>
<br><br>

We introduce a pruning framework on the basis of free-text Knowledge Graph put more focus on candidates relevance to question's intention.
- Two classifiers are trained to indentify the question's intention and summerzie the entity's semantics by label. 
- A pruned free-text KG is generated according to labels. 

# 0 Dependency Installation
Run python setup.py develop to install required dependencies for DELFT.


# 1 Data and trained model Download

For each experimented dataset, we provide processed graph input [here](https://obj.umiacs.umd.edu/delft_release/grounded_graph.zip), after downloading, unzip it and put into data folder (the tokenized version is for glove setting, while the other is for bert setting).
We also provide trained model [here](https://obj.umiacs.umd.edu/delft_release/trained_models.zip), unzip the downloaded model and put into experiments folder. Also, the original datasets are [here](https://obj.umiacs.umd.edu/delft_release/original_questions.zip).

For delft glove, downloading pre-trained glove embeddings are required, the link is [here](http://downloads.cs.stanford.edu/nlp/data/glove.840B.300d.zip).
## 2 Graph pruning
The relevant content for Graph pruning is palced under the folder of BLP, please check for reference.
For the purpose of experiments, we provide pruned free-text KG on TriviaQA and QBLink's test set [here](https://drive.google.com/file/d/1l9rlbQ4sAb6VmRFt7flseKVXG0OIY99e/view?usp=sharing)

## 3 Run Experiments
The experiments include DELFT-Bert, DELFT-Glove and newly introduced DELFT-LUKE, with different embeddings, each experiment has a seperate folder (see readme.md on each folder).


## Contact
Please [email](huishiqiu@gmail.com) to me if you have any issues
