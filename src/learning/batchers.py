"""
Classes to stream int-mapped data from file in batches, pad and sort them (as needed)
and return batch dicts for the models.
"""
import codecs
import pprint, sys
import copy
import random
import itertools
import re
from collections import defaultdict
import logging

import numpy as np
import torch
from transformers import AutoTokenizer

from . import data_utils as du

replace_sep = re.compile(r'\[SEP\]')


class GenericBatcher:
    def __init__(self, num_examples, batch_size):
        """
        Maintain batcher variables, state and such. Any batcher for a specific
        model is a subclass of this and implements specific methods that it
        needs.
        - A batcher needs to know how to read from an int-mapped raw-file.
        - A batcher should yield a dict which you model class knows how to handle.
        :param num_examples: the number of examples in total.
        :param batch_size: the number of examples to have in a batch.
        """
        # Batch sizes book-keeping; the 0 and -1 happen in the case of test time usage.
        if num_examples > 0 and batch_size > -1:
            self.full_len = num_examples
            self.batch_size = batch_size
            if self.full_len > self.batch_size:
                self.num_batches = int(np.ceil(float(self.full_len) / self.batch_size))
            else:
                self.num_batches = 1
    
            # Get batch indices.
            self.batch_start = 0
            self.batch_end = self.batch_size

    def next_batch(self):
        """
        This should yield the dict which your model knows how to make sense of.
        :return:
        """
        raise NotImplementedError

    @staticmethod
    def raw_batch_from_file(ex_file, to_read_count):
        """
        Implement whatever you need for reading a raw batch of examples.
        Read the next batch from the file.
        :param ex_file: File-like with a next() method.
        :param to_read_count: int; number of examples to read from the file.
        :return:
        """
        raise NotImplementedError


class SentTripleBatcher(GenericBatcher):
    """
    Feeds a model which inputs query, positive. Negatives are in-batch.
    """
    bert_config_str = None
    
    def __init__(self, ex_fnames, num_examples, batch_size):
        """
        Batcher class for the em style trained models.
        This batcher is also used at test time, at this time all the arguments here are
        meaningless. Only the make_batch and ones beneath it will be used.
        :param ex_fnames: dict('pos_ex_fname': str, 'neg_ex_fname': str)
        :param num_examples: int.
        :param batch_size: int.
        :param bert_config: string; BERT config string to initialize tokenizer with.
        :param max_pos_neg: int; maximum number of positive and negative examples per
            query to train with.
        """
        GenericBatcher.__init__(self, num_examples=num_examples,
                                batch_size=batch_size)
        # Call it pos ex fname even so code elsewhere can be re-used.
        if ex_fnames:
            pos_ex_fname = ex_fnames['pos_ex_fname']
            # Access the file with the sentence level examples.
            self.pos_ex_file = codecs.open(pos_ex_fname, 'r', 'utf-8')
        self.pt_lm_tokenizer = AutoTokenizer.from_pretrained(self.bert_config_str)
    
    def next_batch(self):
        """
        Yield the next batch. Based on whether its train_mode or not yield a
        different set of items.
        :return:
            batch_doc_ids: list; with the doc_ids corresponding to the
                    examples in the batch.
            batch_dict: see make_batch.
        """
        for nb in range(self.num_batches):
            # Read the batch of data from the file.
            if self.batch_end < self.full_len:
                cur_batch_size = self.batch_size
            else:
                cur_batch_size = self.full_len - self.batch_start
            batch_query_docids, batch_queries, batch_pos, batch_neg = \
                next(SentTripleBatcher.raw_batch_from_file(self.pos_ex_file, cur_batch_size))
            self.batch_start = self.batch_end
            self.batch_end += self.batch_size
            try:
                if batch_neg and batch_pos:
                    feed = {'query_texts': batch_queries, 'pos_texts': batch_pos, 'neg_texts': batch_neg}
                elif batch_pos:
                    feed = {'query_texts': batch_queries, 'pos_texts': batch_pos}
                else:
                    feed = {'query_texts': batch_queries}
                batch_dict = self.make_batch(raw_feed=feed, pt_lm_tokenizer=self.pt_lm_tokenizer)
            except (IndexError, AssertionError) as error:
                print(batch_query_docids)
                print(batch_queries)
                print(batch_pos)
                sys.exit()
            batch_dict = {
                'batch_rank': batch_dict
            }
            yield batch_query_docids, batch_dict
    
    @staticmethod
    def raw_batch_from_file(ex_file, to_read_count):
        """
        Read the next batch from the file. In reading the examples:
        - For every query only read max_pos_neg positive and negative examples.
        :param ex_file: File-like with a next() method.
        :param to_read_count: int; number of lines to read from the file.
        :return:
            query_abs: list(str); list of query sentences
            pos_abs: list(str); list of positive sentences
            neg_abs: list(str); list of negative sentences
        """
        # Initial values.
        read_ex_count = 0
        # These will be to_read_count long.
        ex_query_docids = []
        query_texts = []
        pos_texts = []
        neg_texts = []
        # Read content from file until the file content is exhausted.
        for ex in du.read_json(ex_file):
            docids = read_ex_count
            ex_query_docids.append(docids)
            query_texts.append(ex['query'])
            # Dont assume even a positive is present -- happens because of
            # SimCSE like pretraining.
            try:
                pos_texts.append(ex['pos_context'])
            except KeyError:
                pass
            # Only dev files have neg examples. Pos used inbatch negs.
            try:
                neg_texts.append(ex['neg_context'])
            except KeyError:
                pass
            read_ex_count += 1
            if read_ex_count == to_read_count:
                yield ex_query_docids, query_texts, pos_texts, neg_texts
                # Once execution is back here empty the lists and reset counters.
                read_ex_count = 0
                ex_query_docids = []
                query_texts = []
                pos_texts = []
                neg_texts = []
    
    @staticmethod
    def make_batch(raw_feed, pt_lm_tokenizer):
        """
        Creates positive and query batches. Only used for training. Test use happens
        with embeddings generated in the pre_proc_poolbaselines scripts.
        :param raw_feed: dict; a dict with the set of things you want to feed
            the model.
        :return:
            batch_dict: dict of the form:
            {
                'query_bert_batch': dict(); The batch which BERT inputs with query sents;
                    Tokenized and int mapped sentences and other inputs to BERT.
                'pos_bert_batch': dict();  The batch which BERT inputs with positive sents;
                    Tokenized and int mapped sentences and other inputs to BERT.
            }
        """
        # Unpack arguments.
        query_texts = raw_feed['query_texts']
        pos_texts = raw_feed['pos_texts']
        # Get bert batches and prepare sep token indices.
        qbert_batch, _, _ = SentTripleBatcher.prepare_bert_sentences(sents=query_texts, tokenizer=pt_lm_tokenizer)
        pbert_batch, _, _ = SentTripleBatcher.prepare_bert_sentences(sents=pos_texts, tokenizer=pt_lm_tokenizer)

        # Happens with the dev set in models using triple losses and in batch negs.
        if 'neg_texts' in raw_feed:
            neg_texts = raw_feed['neg_texts']
            nbert_batch, _, _ = SentTripleBatcher.prepare_bert_sentences(sents=neg_texts, tokenizer=pt_lm_tokenizer)
            batch_dict = {
                'query_bert_batch': qbert_batch,
                'pos_bert_batch': pbert_batch,
                'neg_bert_batch': nbert_batch
            }
        else:
            batch_dict = {
                'query_bert_batch': qbert_batch,
                'pos_bert_batch': pbert_batch
            }
        return batch_dict
    
    @staticmethod
    def prepare_bert_sentences(sents, tokenizer):
        """
        Given a batch of sentences prepare a batch which can be passed through BERT.
        :param sents: list(string)
        :param tokenizer: an instance of the appropriately initialized BERT tokenizer.
        :return:
        """
        max_num_toks = 500
        # Construct the batch.
        tokenized_batch = []
        tokenized_text = []
        batch_seg_ids = []
        batch_attn_mask = []
        seq_lens = []
        max_seq_len = -1
        for sent in sents:
            bert_tokenized_text = tokenizer.tokenize(sent)
            bert_tokenized_text = bert_tokenized_text[:max_num_toks]
            tokenized_text.append(bert_tokenized_text)
            # Convert token to vocabulary indices
            indexed_tokens = tokenizer.convert_tokens_to_ids(bert_tokenized_text)
            # Append CLS and SEP tokens to the text..
            indexed_tokens = tokenizer.build_inputs_with_special_tokens(token_ids_0=indexed_tokens)
            if len(indexed_tokens) > max_seq_len:
                max_seq_len = len(indexed_tokens)
            seq_lens.append(len(indexed_tokens))
            tokenized_batch.append(indexed_tokens)
            batch_seg_ids.append([0] * len(indexed_tokens))
            batch_attn_mask.append([1] * len(indexed_tokens))
        # Pad the batch.
        for ids_sent, seg_ids, attn_mask in zip(tokenized_batch, batch_seg_ids, batch_attn_mask):
            pad_len = max_seq_len - len(ids_sent)
            ids_sent.extend([tokenizer.pad_token_id] * pad_len)
            seg_ids.extend([tokenizer.pad_token_id] * pad_len)
            attn_mask.extend([tokenizer.pad_token_id] * pad_len)
        # The batch which the BERT model will input.
        bert_batch = {
            'tokid_tt': torch.tensor(tokenized_batch),
            'seg_tt': torch.tensor(batch_seg_ids),
            'attnmask_tt': torch.tensor(batch_attn_mask),
            'seq_lens': seq_lens
        }
        return bert_batch, tokenized_text, tokenized_batch


class AbsTripleBatcher(SentTripleBatcher):
    @staticmethod
    def make_batch(raw_feed, pt_lm_tokenizer):
        """
        Creates positive and query batches. Only used for training. Test use happens
        with embeddings generated in the pre_proc_poolbaselines scripts.
        :param raw_feed: dict; a dict with the set of things you want to feed
            the model.
        :return:
            batch_dict: dict of the form:
            {
                'query_bert_batch': dict(); The batch which BERT inputs with query sents;
                    Tokenized and int mapped sentences and other inputs to BERT.
                'pos_bert_batch': dict();  The batch which BERT inputs with positive sents;
                    Tokenized and int mapped sentences and other inputs to BERT.
            }
        """
        # Unpack arguments.
        query_texts = raw_feed['query_texts']
        # Get bert batches and prepare sep token indices.
        qbert_batch = AbsTripleBatcher.prepare_abstracts(batch_abs=query_texts, pt_lm_tokenizer=pt_lm_tokenizer)

        # Happens with the dev set in models using triple losses and in batch negs.
        if 'neg_texts' in raw_feed and 'pos_texts' in raw_feed:
            neg_texts = raw_feed['neg_texts']
            nbert_batch = AbsTripleBatcher.prepare_abstracts(batch_abs=neg_texts, pt_lm_tokenizer=pt_lm_tokenizer)
            pos_texts = raw_feed['pos_texts']
            pbert_batch = AbsTripleBatcher.prepare_abstracts(batch_abs=pos_texts, pt_lm_tokenizer=pt_lm_tokenizer)
            batch_dict = {
                'query_bert_batch': qbert_batch,
                'pos_bert_batch': pbert_batch,
                'neg_bert_batch': nbert_batch
            }
        # Happens at train when using in batch negs.
        elif 'pos_texts' in raw_feed:
            pos_texts = raw_feed['pos_texts']
            pbert_batch = AbsTripleBatcher.prepare_abstracts(batch_abs=pos_texts, pt_lm_tokenizer=pt_lm_tokenizer)
            batch_dict = {
                'query_bert_batch': qbert_batch,
                'pos_bert_batch': pbert_batch
            }
        # Happens when the function is called from other scripts to encode text.
        else:
            batch_dict = {
                'bert_batch': qbert_batch,
            }
        return batch_dict

    @staticmethod
    def prepare_abstracts(batch_abs, pt_lm_tokenizer):
        """
        Given the abstracts sentences as a list of strings prep them to pass through model.
        :param batch_abs: list(dict); list of example dicts with sentences, facets, titles.
        :return:
            bert_batch: dict(); returned from prepare_bert_sentences.
        """
        # Prepare bert batch.
        batch_abs_seqs = []
        # Add the title and abstract concated with seps because thats how SPECTER did it.
        for ex_abs in batch_abs:
            assert(isinstance(ex_abs['TITLE'], str))  # title is a string.
            assert(isinstance(ex_abs['ABSTRACT'], list))  # abstract is a list of strings.
            seqs = [ex_abs['TITLE'] + ' [SEP] ']
            seqs.extend(ex_abs['ABSTRACT'])
            batch_abs_seqs.append(' '.join(seqs))
        bert_batch, tokenized_abs, tokenized_ids = SentTripleBatcher.prepare_bert_sentences(
            sents=batch_abs_seqs, tokenizer=pt_lm_tokenizer)
        return bert_batch
