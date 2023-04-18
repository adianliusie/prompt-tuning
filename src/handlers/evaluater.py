import torch
import pickle
import os
import sacrebleu
import numpy as np
import torch.nn.functional as F

from rouge_score import rouge_scorer
from tqdm import tqdm 
from types import SimpleNamespace

from .trainer import Trainer
from ..data.data_handler import DataHandler

from ..utils.torch import set_rand_seed

class Evaluater(Trainer):
    """ Evaluator class- inherits Trainer so has all experiment methods
        class takes care of evaluation and automatic caching of results"""

    def __init__(self, path:str, device:str='cuda'):
        self.exp_path = path
        self.device = device

    def setup_helpers(self):
        # load arguments 
        args = self.load_args('model_args.json')

        # set up attributes 
        super().setup_helpers(args)

        # load model weights
        self.load_model()
            
    #== loading and saving generated texts =========================================================#
    def load_pred_texts(self, dataset:str, mode:str='test', template=None):
        if not self.texts_exist(dataset, mode, template):
            self.setup_helpers()
            texts = self.generate_texts(dataset, mode, template)
            self.save_cache_texts(texts, dataset, mode, template)
        texts = self.load_cached_texts(dataset, mode, template)
        return texts

    def save_cache_texts(self, texts:dict, dataset:str, mode:str, template=None):
        text_cache_path = self.get_text_cache_path(dataset, mode, template)
        with open(text_cache_path, 'wb') as handle:
            pickle.dump(texts, handle)

    def load_cached_texts(self, dataset:str, mode:str, template=None)->dict:
        text_cache_path = self.get_text_cache_path(dataset, mode, template)
        with open(text_cache_path, 'rb') as handle:
            texts = pickle.load(handle)
        return texts

    def texts_exist(self, dataset:str, mode:str, template=None)->bool:
        text_cache_path = self.get_text_cache_path(dataset, mode, template)
        return os.path.isfile(text_cache_path)

    def get_text_cache_path(self, dataset:str, mode:str, template=None)->str:
        # get cache name dependening on parameters
        if template == None:
            eval_name = f"{dataset}_{mode}"
        else:
            eval_name = f"{dataset}_{mode}_{template.replace(' ', '-')}"

        # load absolute path base on exp
        text_cache_path = os.path.join(self.exp_path, 'eval', f'{eval_name}.pk')
        return text_cache_path

    #== Model probability calculation method ======================================================#
    @torch.no_grad()
    def generate_texts(
        self, 
        dataset:str, 
        mode:str='test', 
        template=None, 
        num_beams:int=4, 
        lim:int=None
    ):
        self.model.eval()
        self.to(self.device)
        set_rand_seed(1)

        # set template if set
        if template != None:
            self.data_handler.template = template
            
        eval_data = self.data_handler.prep_split(dataset, mode, lim)
        eval_batches = self.batcher(
            data = eval_data,
            bsz = 1,
            shuffle = False
        )
        output = {}
        for batch in tqdm(eval_batches, total=len(eval_data)):
            ex_id = batch.ex_id[0]

            # Generate free running prediction
            free_output = self.model.generate(
                input_ids = batch.input_ids,
                attention_mask = batch.attention_mask,
                max_length = self.batcher.max_len,
                num_beams = num_beams,
                length_penalty = 0.6,
                no_repeat_ngram_size = 4,
                num_return_sequences = 1,
                output_scores = True,
                return_dict_in_generate = True,
            )

            beams = free_output.sequences
            texts = self.data_handler.tokenizer.batch_decode(beams, skip_special_tokens=True)
            
            output[ex_id] = texts[0]
        return output

    #== Teacher Forcing LogProbability distribution ===============================================#
    def load_likelihood(self, dataset:str, mode:str='test', template=None):
        if not self.likelihood_exists(dataset, mode, template):
            self.setup_helpers()
            metrics = self.calculate_likelihood(dataset, mode, template)
            self.save_likelihood(metrics, dataset, mode, template)
        metrics = self.load_cached_likelihood(dataset, mode, template)
        return metrics

    def calculate_likelihood(self, dataset:str, mode:str='test', template=None, lim:int=None):
        if template != None:
            self.data_handler.template = template
            
        eval_data = self.data_handler.prep_split(dataset, mode, lim)
        self.to(self.device)
        metrics = self.run_validation(eval_data, mode='test', tqdm_display=True)
        return metrics
    
    def save_likelihood(self, texts:dict, dataset:str, mode:str, template:str=None):
        likelihood_cache_path = self.get_likelihood_cache_path(dataset, mode, template)
        with open(likelihood_cache_path, 'wb') as handle:
            pickle.dump(texts, handle)

    def load_cached_likelihood(self, dataset:str, mode:str, template:str=None)->dict:
        likelihood_cache_path = self.get_likelihood_cache_path(dataset, mode, template)
        with open(likelihood_cache_path, 'rb') as handle:
            texts = pickle.load(handle)
        return texts

    def likelihood_exists(self, dataset:str, mode:str, template:str=None)->bool:
        likelihood_cache_path = self.get_likelihood_cache_path(dataset, mode, template)
        return os.path.isfile(likelihood_cache_path)

    def get_likelihood_cache_path(self, dataset:str, mode:str, template:str=None)->str:
        # get cache name dependening on parameters
        if template == None:
            eval_name = f"TF_{dataset}_{mode}"
        else:
            eval_name = f"TF_{dataset}_{mode}_{template.replace(' ', '-')}"

        # load absolute path base on exp
        text_cache_path = os.path.join(self.exp_path, 'eval', f'{eval_name}.pk')
        return text_cache_path

    #== loading specific examples =================================================================#
    def load_ex(self, dataset:str, mode:str, k:int=0)->SimpleNamespace:
        data = self.data_handler.prep_split(dataset, mode, lim=10) #TEMP
        ex = data[k]
        return ex
    
    def tokenize_ex(self, ex:SimpleNamespace):
        ex = self.data_handler._prep_ids([ex])
        batch = next(self.batcher(ex, bsz=1))
        return batch
    
    #== general eval methods ======================================================================#
    @staticmethod
    def load_label_texts(dataset:str, mode:str='test')->dict:
        eval_data = DataHandler.load_split(dataset, mode)
        labels_dict = {}
        for ex in eval_data:
            labels_dict[ex.ex_id] = ex.label_text
        return labels_dict

    @staticmethod
    def load_split(dataset:str, mode:str='test', lim=None)->dict:
        eval_data = DataHandler.load_split(dataset, mode)
        output_dict = {}
        for ex in eval_data:
            output_dict[ex.ex_id] = ex
        return output_dict

    @staticmethod
    def calculate_bleu(pred_texts:dict, label_texts:dict, display:bool=False):
        assert pred_texts.keys() == label_texts.keys()
        
        # format inputs
        ex_ids = label_texts.keys()
        preds_text_list  = [pred_texts[k] for k in ex_ids]
        labels_text_list = [[label_texts[k] for k in ex_ids]]
        
        # calculate bleu score
        bleu_score = sacrebleu.corpus_bleu(
            preds_text_list,
            labels_text_list,
        ).score

        if display:
            print(f'bleu:    {bleu_score:.3f}')

        return bleu_score    

    @staticmethod
    def calculate_rouge(pred_texts:dict, label_texts:dict, display:bool=False):
        assert pred_texts.keys() == label_texts.keys()

        # set up rouge calculation
        rouge1, rouge2, rougeL = [], [], []
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        # calculate rouge for each text, and average over all dataset
        for ex_id in label_texts.keys():
            text_scores = scorer.score(pred_texts[ex_id], label_texts[ex_id])
            rouge1.append(text_scores['rouge1'][2])
            rouge2.append(text_scores['rouge2'][2])
            rougeL.append(text_scores['rougeL'][2])
    
        rouge_1 = np.mean(rouge1)
        rouge_2 = np.mean(rouge2)
        rouge_L = np.mean(rougeL)

        if display:
            print(f'rouge_1: {rouge_1:.3f}')
            print(f'rouge_2: {rouge_2:.3f}')
            print(f'rouge_L: {rouge_L:.3f}')

        return rouge1, rouge2, rougeL

