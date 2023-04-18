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
    prep_wiki_bio([train, dev, test])
    return dev, dev, dev

def load_common_wikibio():
    dataset = load_dataset("wiki_bio")
    data = []
    for split in ['train', 'val', 'test']:
        data = dataset[split]
        sorted_data = data.sort(lambda x: len(x['target_text']), reverse=True)
        data = sorted_data
    train, dev, test = dataset['train'], dataset['val'], dataset['test']
    
def load_small_wikibio():
    """loads 5% of the data"""
    dataset = load_dataset("wiki_bio")
    output_data = []
    
    # load data splits, and only keep 5% of the splits
    for split in ['train', 'val', 'test']:
        data = dataset[split]
        temp_split = data.train_test_split(test_size=0.05)
        data = temp_split['test']
        output_data.append(data)
        
    # format the data for the framework
    train, dev, test = output_data
    train, dev, test = _rename_keys(train, dev, test, old_key='target_text', new_key='label_text')
    prep_wiki_bio([train, dev, test])
    
    return train, dev, test

    
def prep_wiki_bio(splits):
    for data in splits:
        for ex in data:
            ex['name'] = ex['input_text']['context'].replace('\n', '')
        
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
