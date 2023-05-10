"""
Biencoder models.
"""
import collections
import logging
import os, codecs, json
import numpy as np
import torch
from torch import nn as nn
from torch.autograd import Variable
from transformers import AutoModel

from ..models_common import generic_layers as gl


class MyBiencoder(nn.Module):
    """
    Pass abstract through SciBERT all in one shot, read off cls token and use
    it to compute similarities. This is an unfaceted model and is meant to
    be similar to SPECTER in all aspects:
    - triplet loss function
    - only final layer cls bert representation
    - no SEP tokens in between abstract sentences
    """
    
    def __init__(self, model_hparams, bert_config=None):
        """
        :param model_hparams: dict(string:int); model hyperparams.
            num_code_vecs: int; number of code vectors to disentangle into.
                The number of facets.
            num_tf_heads: int; number of heads in the context transformer.
        :param bert_config: transformers.configuration_bert.BertConfig; bert
            hyperparam instance.
        """
        torch.nn.Module.__init__(self)
        self.bert_config = bert_config
        self.bert_encoding_dim = 768  # bert_config.hidden_size or DistilBertConfig.dim
        self.bert_layer_count = 12 + 1  # plus 1 for the bottom most layer.
        self.bert_encoder = AutoModel.from_pretrained(model_hparams['base-pt-layer'])
        self.bert_encoder.config.output_hidden_states = True
        # If fine tune is False then freeze the bert params.
        if not model_hparams['fine_tune']:
            for param in self.bert_encoder.base_model.parameters():
                param.requires_grad = False
        self.bert_layer_weights = gl.SoftmaxMixLayers(in_features=self.bert_layer_count, out_features=1, bias=False)
        self.criterion = nn.TripletMarginLoss(margin=1, p=2, reduction='sum')
        if 'consent-base-pt-layer' in model_hparams:  # if this is passed get all model params from there.
            self.load_state_dict(load_aspire_model(expanded_model_name=model_hparams['consent-base-pt-layer'],
                                                   get_whole_state_dict=True))

    def caching_score(self, query_encode_ret_dict, cand_encode_ret_dicts):
        """
        Called externally from a class using the trained model.
        - Create as many repetitions of query_reps as cand_reps.
        - Compute scores and return.
        query_encode_ret_dict: list({'sent_reps': numpy.array, 'doc_cls_reps': numpy.array})
        cand_encode_ret_dict: list({'sent_reps': numpy.array, 'doc_cls_reps': numpy.array})
        """
        # Pack representations as padded gpu tensors.
        query_cls_reps = [d['doc_cls_reps'] for d in query_encode_ret_dict]
        num_query_abs = len(query_cls_reps)
        query_cls_reps = np.vstack(query_cls_reps)
        cand_cls_reps = [d['doc_cls_reps'] for d in cand_encode_ret_dicts]
        batch_size = len(cand_cls_reps)
        flat_query_cls_reps = np.zeros((batch_size, num_query_abs, self.bert_encoding_dim))
        for bi in range(batch_size):
            flat_query_cls_reps[bi, :num_query_abs, :] = query_cls_reps
        flat_query_cls_reps, cand_cls_reps = Variable(torch.FloatTensor(flat_query_cls_reps)),\
                                             Variable(torch.FloatTensor(np.vstack(cand_cls_reps)))
        if torch.cuda.is_available():
            # batch_size x num_query_abs x encoding_dim
            flat_query_cls_reps = flat_query_cls_reps.cuda()
            # batch_size x encoding_dim
            cand_cls_reps = cand_cls_reps.cuda()
        # Compute scores from all user docs to candidate docs.
        cand2user_doc_sims = -1*torch.cdist(cand_cls_reps.unsqueeze(1), flat_query_cls_reps)
        # batch_size x 1 x num_query_abs
        cand_sims, _ = torch.max(cand2user_doc_sims.squeeze(1), dim=1)
        # Make numpy arrays and return.
        if torch.cuda.is_available():
            batch_scores = cand_sims.cpu().data.numpy()
        else:
            batch_scores = cand_sims.data.numpy()
        # Return the same thing as batch_scores and pair_scores because the pp_gen_nearest class expects it.
        ret_dict = {
            'batch_scores': batch_scores,
            'pair_scores': batch_scores
        }
        return ret_dict

    def caching_encode(self, batch_dict):
        """
        Function used at test time.
        batch_dict: dict of the form accepted by forward_rank but without any of the
            negative examples.
        :return: ret_dict
        """
        doc_bert_batch, batch_size = batch_dict['bert_batch'], len(batch_dict['bert_batch']['seq_lens'])
        # Get the representations from the model; batch_size x encoding_dim x max_sents
        doc_cls_reps = self.partial_forward(bert_batch=doc_bert_batch)
        # Make numpy arrays and return.
        if torch.cuda.is_available():
            doc_cls_reps = doc_cls_reps.cpu().data.numpy()
        else:
            doc_cls_reps = doc_cls_reps.data.numpy()
        # Return a list of reps instead of reps collated as one np array.
        batch_reps = []
        for i in range(batch_size):
            batch_reps.append({'doc_cls_reps': doc_cls_reps[i, :]})
        return batch_reps

    def encode(self, batch_dict):
        """
        Function used at test time.
        batch_dict: dict of the form accepted by forward_rank but without any of the
            negative examples.
        :return: ret_dict
        """
        doc_bert_batch = batch_dict['bert_batch']
        # Get the representations from the model.
        doc_reps = self.partial_forward(bert_batch=doc_bert_batch)
        # Make numpy arrays and return.
        if torch.cuda.is_available():
            doc_reps = doc_reps.cpu().data.numpy()
        else:
            doc_reps = doc_reps.data.numpy()
        ret_dict = {
            'doc_reps': doc_reps,  # batch_size x encoding_dim
        }
        return ret_dict

    def forward(self, batch_dict):
        batch_loss = self.forward_rank(batch_dict['batch_rank'])
        loss_dict = {
            'rankl': batch_loss
        }
        return loss_dict

    def forward_rank(self, batch_rank):
        """
        Function used at training time.
        batch_dict: dict of the form:
            {
                'query_bert_batch': dict(); The batch which BERT inputs with flattened and
                    concated sentences from query abstracts; Tokenized and int mapped
                    sentences and other inputs to BERT.
                'pos_bert_batch': dict(); The batch which BERT inputs with flattened and
                    concated sentences from positive abstracts; Tokenized and int mapped
                    sentences and other inputs to BERT.
                'neg_bert_batch': dict(); The batch which BERT inputs with flattened and
                    concated sentences from query abstracts; Tokenized and int mapped
                    sentences and other inputs to BERT.
            }
        :return: loss_val; torch Variable.
        """
        qbert_batch = batch_rank['query_bert_batch']
        pbert_batch = batch_rank['pos_bert_batch']
        # Get the representations from the model.
        q_sent_reps = self.partial_forward(bert_batch=qbert_batch)
        p_context_reps = self.partial_forward(bert_batch=pbert_batch)
        # Happens when running on the dev set.
        if 'neg_bert_batch' in batch_rank:
            nbert_batch = batch_rank['neg_bert_batch']
            n_context_reps = self.partial_forward(bert_batch=nbert_batch)
        else:
            # Use a shuffled set of positives as the negatives. -- in-batch negatives.
            n_context_reps = p_context_reps[torch.randperm(p_context_reps.size()[0])]
        loss_val = self.criterion(q_sent_reps, p_context_reps, n_context_reps)
        return loss_val
    
    def partial_forward(self, bert_batch):
        """
        Function shared between the training and test time behaviour. Pass a batch
        of sentences through BERT and return cls representations.
        :return:
            cls_doc_reps: batch_size x encoding_dim
        """
        # batch_size x bert_encoding_dim
        cls_doc_reps = self.doc_reps_bert(bert_batch=bert_batch)
        if len(cls_doc_reps.size()) == 1:
            cls_doc_reps = cls_doc_reps.unsqueeze(0)
        return cls_doc_reps

    def doc_reps_bert(self, bert_batch):
        """
        Pass the concated abstract through BERT, and read off [SEP] token reps to get sentence reps,
        and weighted combine across layers.
        :param bert_batch: dict('tokid_tt', 'seg_tt', 'attnmask_tt', 'seq_lens'); items to use for getting BERT
            representations. The sentence mapped to BERT vocab and appropriately padded.
        :return:
            doc_cls_reps: FloatTensor [batch_size x bert_encoding_dim]
        """
        tokid_tt, seg_tt, attnmask_tt = bert_batch['tokid_tt'], bert_batch['seg_tt'], bert_batch['attnmask_tt']
        if torch.cuda.is_available():
            tokid_tt, seg_tt, attnmask_tt = tokid_tt.cuda(), seg_tt.cuda(), attnmask_tt.cuda()
        # Pass input through BERT and return all layer hidden outputs.
        model_outputs = self.bert_encoder(tokid_tt, token_type_ids=seg_tt, attention_mask=attnmask_tt)
        # Weighted combine the hidden_states which is a list of [bs x max_seq_len x bert_encoding_dim]
        # with as many tensors as layers + 1 input layer.
        hs_stacked = torch.stack(model_outputs.hidden_states, dim=3)
        weighted_sum_hs = self.bert_layer_weights(hs_stacked)  # [bs x max_seq_len x bert_encoding_dim x 1]
        weighted_sum_hs = torch.squeeze(weighted_sum_hs, dim=3)
        # Read of CLS token as document representation: (batch_size, sequence_length, hidden_size)
        cls_doc_reps = weighted_sum_hs[:, 0, :]
        cls_doc_reps = cls_doc_reps.squeeze()
        return cls_doc_reps


class BiEncoderAv(MyBiencoder):
    """
    Pass abstract through BERT all in one shot, average the tokens
    maximize L2 distances bw
    """
    def __init__(self, model_hparams, bert_config=None):
        """
        :param model_hparams: dict(string:int); model hyperparams.
            num_code_vecs: int; number of code vectors to disentangle into.
                The number of facets.
            num_tf_heads: int; number of heads in the context transformer.
        :param bert_config: transformers.configuration_bert.BertConfig; bert
            hyperparam instance.
        """
        torch.nn.Module.__init__(self)
        self.bert_config = bert_config
        self.bert_encoding_dim = 768  # bert_config.hidden_size or DistilBertConfig.dim
        self.bert_layer_count = 12 + 1  # plus 1 for the bottom most layer.
        self.bert_encoder = AutoModel.from_pretrained(model_hparams['base-pt-layer'])
        self.bert_encoder.config.output_hidden_states = True
        # If fine tune is False then freeze the bert params.
        if not model_hparams['fine_tune']:
            for param in self.bert_encoder.base_model.parameters():
                param.requires_grad = False
        self.criterion = nn.TripletMarginLoss(margin=1, p=2, reduction='sum')
    
    def doc_reps_bert(self, bert_batch):
        """
        Pass the document through BERT and average the token reps to
        get document embeddings.
        :param bert_batch: dict('tokid_tt', 'seg_tt', 'attnmask_tt', 'seq_lens'); items to use
            for getting BERT representations. The sentence mapped to BERT vocab and
            appropriately padded.
        :return:
            doc_cls_reps: FloatTensor [batch_size x bert_encoding_dim]
        """
        seq_lens = bert_batch['seq_lens']
        batch_size, max_seq_len = len(seq_lens), max(seq_lens)
        tok_mask = np.zeros((batch_size, max_seq_len, self.bert_encoding_dim))
        # Build a mask for
        for i, seq_len in enumerate(seq_lens):
            tok_mask[i, :seq_len, :] = 1.0
        tokid_tt, seg_tt, attnmask_tt = bert_batch['tokid_tt'], bert_batch['seg_tt'], bert_batch['attnmask_tt']
        tok_mask = Variable(torch.FloatTensor(tok_mask))
        if torch.cuda.is_available():
            tokid_tt, seg_tt, attnmask_tt = tokid_tt.cuda(), seg_tt.cuda(), attnmask_tt.cuda()
            tok_mask = tok_mask.cuda()
        # Pass input through BERT and return all layer hidden outputs.
        model_outputs = self.bert_encoder(tokid_tt, token_type_ids=seg_tt, attention_mask=attnmask_tt)
        final_hidden_state = model_outputs.last_hidden_state
        doc_tokens = final_hidden_state * tok_mask
        # The sent_masks non zero elements in one slice along embedding dim is the sentence length.
        dov_reps_av = torch.sum(doc_tokens, dim=1) / torch.count_nonzero(
            tok_mask[:, :, 0], dim=1).clamp(min=1).unsqueeze(dim=1)
        return dov_reps_av


