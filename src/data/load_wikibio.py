import random

from copy import deepcopy
from typing import List, Dict, Tuple, TypedDict
from datasets import load_dataset

class Seq2seqText(TypedDict):
    """Output example formatting (only here for documentation)"""
    input_text : str
    label_text : int

#== main loading function =========================================================================# 
def load_wikibio():
    dataset = load_dataset("wiki_bio")
    train, dev, test = dataset['train'], dataset['val'], dataset['test']
    train, dev, test = _rename_keys(train, dev, test, old_key='target_text', new_key='label_text')
    return train, dev, test

def _rename_keys(train:list, dev:list, test:list, old_key:str, new_key:str):
    train = [_rename_key(ex, old_key, new_key) for ex in train]
    dev   = [_rename_key(ex, old_key, new_key) for ex in dev]
    test  = [_rename_key(ex, old_key, new_key) for ex in test]
    return train, dev, test

def _rename_key(ex:dict, old_key:str='content', new_key:str='text'):
    """ convert key name from the old_key to 'text' """
    ex = ex.copy()
    ex[new_key] = ex.pop(old_key)
    return ex
