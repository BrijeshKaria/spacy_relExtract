#!/usr/bin/env python
# coding: utf-8

# In[46]:


from __future__ import unicode_literals, print_function

import plac
import random
from pathlib import Path
from spacy.util import minibatch, compounding

import re 
import string 
import nltk 
import spacy 
import pandas as pd 
import numpy as np 
import math 
from tqdm import tqdm 

from spacy.matcher import Matcher 
from spacy.tokens import Span 
from spacy import displacy 

# load spaCy model
nlp = spacy.load("en_core_web_sm")


# In[47]:


Trainingdata="""get me documents with publish date greater than 10-Oct-2010
get me companies with revenue greater than 500000
companies with revenue above 510000
companies with revenue in access of 500000
companies with revenue exceeding 500000
get me companies with revenue less than 500000
get me companies with revenue lesser than $ 5b
get me companies with revenue lesser than five billion
what deals have revenue higher than $5b
get me companies with revenue more than 500000
Which are the companies having revenues of more than 1000000
deals with deal size more than 200000"""
trainingdata_semantics = """ROOT, -, -, -, FIELD, FIELD, GTR, GTR, VAL
ROOT, -, -, -, FIELD, GTR, GTR, VAL
ROOT, -, FIELD, GTR, VAL
ROOT, -, FIELD, GTR, GTR, GTR, VAL
ROOT, -, FIELD, GTR, VAL
ROOT, -, -, -, FIELD, LSR, LSR, VAL
ROOT, -, -, -, FIELD, LSR, LSR, VAL, VAL
ROOT, -, -, -, FIELD, LSR, LSR, VAL, VAL
-, -, ROOT, FIELD, GTR, GTR, VAL, VAL
ROOT, -, -, -, FIELD, GTR, GTR, VAL
-, ROOT, -, -, -, FIELD, -, GTR, GTR, VAL
ROOT, -, FIELD, FIELD, GTR, GTR, VAL"""


# In[48]:


#Print heads
lines = Trainingdata.split('\n')
sem_lines = trainingdata_semantics.split('\n')

training_data=[]
sample_data=()
k=1
j=0
for text in lines:
    doc = nlp(text)
    print("***************************")
    i = 0
    head = {}
    list = []
    dep = []
    toks = []
    sem_deps = sem_lines[j].split(',')
    j+=1
    #i=1
    for tok in doc:
        #print(tok.head.idx)
        list.append(tok.head.i)
        #print(tok.dep_)
        tokdep= str(tok.dep_) 
        #print(tokdep)
        dep.append(sem_deps[i].strip())  ##tokdep)
        i+=1
        toks.append(tok.text)
        #print(tok.text,"-->",tok.dep_,"-->",tok.pos_, tok.head, tok.lefts, tok.tag_)
        #print([w for w in tok.lefts])
    head["heads"]=list
    head["deps"]=dep
    #head["toks"]=toks
    print(k)
    print(head)
    k+=1
    sample_data=(text, head)
    #print(sample_data)
    training_data.append(sample_data)
print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
print(training_data)
TRAIN_DATA=training_data
    


# In[49]:


@plac.annotations(
    model=("Model name. Defaults to blank 'en' model.", "option", "m", str),
    output_dir=("Optional output directory", "option", "o", Path),
    n_iter=("Number of training iterations", "option", "n", int),
)
def main(model=None, output_dir=None, n_iter=15):
    """Load the model, set up the pipeline and train the parser."""
    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank("en")  # create blank Language class
        print("Created blank 'en' model")

    # We'll use the built-in dependency parser class, but we want to create a
    # fresh instance – just in case.
    print(nlp.pipe_names)
    if "parser" in nlp.pipe_names:
        print("Removing parser")
        nlp.remove_pipe("parser")
    parser = nlp.create_pipe("parser")
    print("Add to pipeline")
    nlp.add_pipe(parser, first=True)

    for text, annotations in TRAIN_DATA:
        for dep in annotations.get("deps", []):
            print(dep)
            parser.add_label(dep)

    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "parser"]
    with nlp.disable_pipes(*other_pipes):  # only train parser
        optimizer = nlp.begin_training()
        for itn in range(n_iter):
            random.shuffle(TRAIN_DATA)
            losses = {}
            # batch up the examples using spaCy's minibatch
            batches = minibatch(TRAIN_DATA, size=compounding(4.0, 32.0, 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)
                print(texts, annotations)
                nlp.update(texts, annotations, sgd=optimizer, losses=losses)
            print("Losses", losses)

    # test the trained model
    test_model(nlp)

    # save model to output directory
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

        # test the saved model
        print("Loading from", output_dir)
        nlp2 = spacy.load(output_dir)
        test_model(nlp2)


def test_model(nlp):
    texts = [
        "deals with size more than 200100",
        "get me companies with sales greater than 400000",
        "companies with revenue in excess of 50000",
    ]
    docs = nlp.pipe(texts)
    for doc in docs:
        print(doc.text)
        print([(t.text, t.dep_, t.head.text) for t in doc if t.dep_ != "-"])


if __name__ == "__main__":
    plac.call(main)

