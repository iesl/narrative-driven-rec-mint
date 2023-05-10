"""
Implement baseline methods which are used for creating pooled papers.
"""
import os
import logging
import re
import time
import codecs, json
import argparse
import collections
import torch
from sklearn import feature_extraction as sk_featext
from sklearn.metrics import pairwise
from transformers import AutoModel, AutoTokenizer
import numpy as np
from numpy import linalg
import scipy.sparse as scisparse
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer, models, CrossEncoder
from rank_bm25 import BM25Okapi

from . import data_utils as du
# from ..learning.facetid_models import disent_models, editie_models
# from ..learning import batchers

np_random_ng = np.random.default_rng()
name2rrmodel = {
        'qlft5base': "google/flan-t5-base",
        'qlft5l': "google/flan-t5-large",
        'qlft5xl': "google/flan-t5-xl",
        'qlft5xxl': "google/flan-t5-xxl",
        'cemsmlm12': "cross-encoder/ms-marco-MiniLM-L-12-v2"
    }


def ql_score_with_t5(model, tokenizer, query_text, poi_texts):
    """
    Given a T5 model rank the poi texts for relevance
    to the query text with query likelihood - we'll filter with this value.
    """
    doc_texts = ["Review: {:s} Write a reddit request asking for recommendations based on this review:".
                 format(d) for d in poi_texts]
    input_encodings = tokenizer(doc_texts, padding='longest', truncation=True, return_tensors='pt')
    input_tok_ids, input_att_mask = input_encodings.input_ids.to('cuda'), input_encodings.attention_mask.to('cuda')
    target_encoding = tokenizer(query_text, truncation=True, return_tensors='pt')
    target_tok_ids = target_encoding.input_ids.to('cuda')
    repeated_target_ids = torch.repeat_interleave(target_tok_ids, len(doc_texts), dim=0)
    # Pass through the model
    with torch.no_grad():
        outs = model(input_ids=input_tok_ids, attention_mask=input_att_mask,
                     labels=repeated_target_ids)
    logits = outs.logits
    log_softmax = torch.nn.functional.log_softmax(logits, dim=2)
    nll = -log_softmax.gather(2, repeated_target_ids.unsqueeze(2)).squeeze(2)
    seq_nll = torch.sum(nll, dim=1)
    doc_text_nlls = seq_nll.cpu().tolist()
    # print(torch.cuda.mem_get_info())
    return doc_text_nlls


class QueryLikelihoodRR:
    """
    Setup a reranking model and rank with it.
    """
    def __init__(self, short_model_name):
        self.tokenizer = T5Tokenizer.from_pretrained(name2rrmodel[short_model_name])
        self.model = T5ForConditionalGeneration.from_pretrained(name2rrmodel[short_model_name], device_map="auto",
                                                                cache_dir='/gypsum/work1/mccallum/smysore/tmp')
    
    def reranking_scores(self, query_text, cand_texts, batch_size=48):
        """
        Return a series of similarity scores (more is better)
        todo: make the standalone function a part of this class and use similarity everywhere
            instead of nll.
        """
        all_log_likelihoods = []
        cand_batch = []
        for cand in cand_texts:
            cand_batch.append(cand)
            if len(cand_batch) == batch_size:
                neg_log_likelihoods = ql_score_with_t5(self.model, self.tokenizer, query_text, cand_batch)
                log_likelihoods = [-1*nll for nll in neg_log_likelihoods]
                all_log_likelihoods.extend(log_likelihoods)
                cand_batch = []
            
        # Handle the final batch.
        if cand_batch:
            neg_log_likelihoods = ql_score_with_t5(self.model, self.tokenizer, query_text, cand_batch)
            log_likelihoods = [-1 * nll for nll in neg_log_likelihoods]
            all_log_likelihoods.extend(log_likelihoods)
        assert(len(cand_texts) == len(all_log_likelihoods))
        return all_log_likelihoods
    

class CrossEncoderRR:
    """
    Set up a reranking model using a pre-trained cross-encoder and rank with it.
    """
    def __init__(self, short_model_name, trained_model_path=None):
        if short_model_name in name2rrmodel:
            self.model = CrossEncoder(name2rrmodel[short_model_name], max_length=512)
        elif short_model_name in {'recinparsce'}:
            print(f'Loading: {trained_model_path}')
            self.model = CrossEncoder(trained_model_path, max_length=512)
        else:
            raise ValueError('Unknown model: {:s}'.format(short_model_name))
    
    def reranking_scores(self, query_text, cand_texts):
        """
        Return a series of similarity scores (more is better)
        """
        inputs = []
        for ct in cand_texts:
            inputs.append((query_text, ct))
        scores = self.model.predict(inputs, batch_size=32)
        return scores


def rankwith_bm25(data_path, run_path, dataset, ranker_name, cand_method=None,
                  trained_reranker_path=None, reranker_name='', torerank=200):
    """
    Score the candidates for relevance with BM25
    :param data_path: string; path from which to read raw documents and queries.
    :param run_path: string; path to save ranked documents to.
    :param dataset: string; {'pointrec'}
    :param ranker_name: string;
    :param cand_method: string; {'mpnet1b', 'tfidf'}
    :return:
    """
    start = time.time()
    if dataset == 'pointrec':
        pool_print_func = print_one_pool_nearest_neighbours_pointrec
    elif dataset == 'sbs2016':
        pool_print_func = print_one_pool_nearest_neighbours_sbs16
    queries_fname = os.path.join(data_path, f'{dataset}_queries.json')
    if cand_method:
        pool_fname = os.path.join(data_path, f'test-pid2all_anns-{dataset}-{cand_method}.json')
        cands_fname = os.path.join(data_path, f'candid2all_cands-{dataset}-{cand_method}.json')
        readable_dir = os.path.join(run_path, f'{dataset}-{ranker_name}{reranker_name}-{cand_method}-ranked')
        out_fname = os.path.join(run_path,
                                 f'test-pid2pool-{dataset}-{ranker_name}{reranker_name}-{cand_method}-ranked.txt')
    else:
        pool_fname = os.path.join(data_path, f'test-pid2anns-{dataset}.json')
        cands_fname = os.path.join(data_path, f'candid2cands-{dataset}.json')
        readable_dir = os.path.join(run_path, f'{dataset}-{ranker_name}{reranker_name}-ranked')
        out_fname = os.path.join(run_path, f'test-pid2pool-{dataset}-{ranker_name}{reranker_name}-ranked.txt')
        
    with codecs.open(queries_fname, 'r', 'utf-8') as fp:
        qid2queries = json.load(fp)
        qid2int = collections.OrderedDict([(qid, i) for i, qid in enumerate(list(qid2queries.keys()))])
        queries = [qid2queries[qid]['query'] for qid in qid2int.keys()]
        print(f'Read: {fp.name}')
    with codecs.open(pool_fname, 'r', 'utf-8') as fp:
        qid2pool = json.load(fp)
        print(f'Read: {fp.name}')
    with codecs.open(cands_fname, 'r', 'utf-8') as fp:
        candid2cands = json.load(fp)
        candid2int = collections.OrderedDict([(cid, i) for i, cid in enumerate(list(candid2cands.keys()))])
        print(f'Read: {fp.name}')
        cand_docs = [candid2cands[cid]['description'] for cid in candid2int.keys()]
    
    cleaned_docs = [re.sub('[^a-zA-Z0-9 \n\.]', '', s) for s in cand_docs]
    tokenized_corpus = [doc.lower().split() for doc in cleaned_docs]
    cleaned_queries = [re.sub('[^a-zA-Z0-9 \n\.]', '', s) for s in queries]
    tokenized_queries = [doc.lower().split() for doc in cleaned_queries]
    
    bm25 = BM25Okapi(tokenized_corpus)
    if reranker_name:
        if reranker_name in {'qlft5base','qlft5l', 'qlft5xl', 'qlft5xxl'}:
            reranking_model = QueryLikelihoodRR(short_model_name=reranker_name)
        elif reranker_name in {'cemsmlm12'}:
            reranking_model = CrossEncoderRR(short_model_name=reranker_name)
        elif reranker_name in {'recinparsce'}:
            reranking_model = CrossEncoderRR(short_model_name=reranker_name,
                                             trained_model_path=trained_reranker_path)

    # Rank the candidates for every query.
    du.create_dir(readable_dir)
    qid2ranked_cands = dict()
    for qid, pool in qid2pool.items():
        cand_ids = pool['cands']
        cand_rels = pool['relevance_adju']
        tokenizedq = tokenized_queries[qid2int[qid]]
        cand2rel = dict(zip(cand_ids, cand_rels))
        all_doc_scores = bm25.get_scores(tokenizedq)
        cand_scores = [all_doc_scores[candid2int[cid]] for cid in cand_ids]
        sims = [score for score in cand_scores]
        assert (len(sims) == len(cand_ids))
        # Sort by first stage ranker sims.
        ranked_cands = list(sorted(zip(cand_ids, sims), key=lambda t: t[1], reverse=True))
        if reranker_name:
            print(f'Reranking {len(qid2ranked_cands)}: {qid}')
            firststage_ranked_cids = [cid for cid, _ in ranked_cands]
            cand_texts = [candid2cands[cid]['description'] for cid in firststage_ranked_cids[:torerank]]
            cand_scores = reranking_model.reranking_scores(query_text=qid2queries[qid]['query'], cand_texts=cand_texts)
            ranked_cands = sorted(zip(firststage_ranked_cids[:torerank], cand_scores), key=lambda t: t[1], reverse=True)
        qid2ranked_cands[qid] = list(ranked_cands)
        ranked_rels = [cand2rel[cid] for cid, d in ranked_cands]
        # Print the ranked cands per query.
        resfile = codecs.open(os.path.join(readable_dir, f'{qid}-{dataset}-{ranker_name}{reranker_name}-ranked.txt'), 'w', 'utf-8')
        pool_print_func(query_id=qid, qid2queries=qid2queries, ranked_cands=ranked_cands,
                        cid2cand_meta=candid2cands, resfile=resfile, cand_relevances=ranked_rels)
        resfile.close()
    # Write results out for trec_eval to read and eval.
    outf = codecs.open(out_fname, 'w', 'utf-8')
    write_trec_eval_out(outfile=outf, qid2ranked_cands=qid2ranked_cands, model_name=f'{ranker_name}{reranker_name}',
                        to_write=200)
    print(f'Wrote: {outf.name}')
    print('Took: {:.4f}s'.format(time.time()-start))
    outf.close()


def cosine_sim(q, d):
    """
    Given two sets of vectors return the cosine distance.
    :param q: np.array(1, dim)
    :param d: np.array(n_samples, dim)
    """
    if len(q.shape) == 1:
        q = q[np.newaxis, :]
    cosine = pairwise.cosine_similarity(q, d)
    return cosine


def l2_sim(q, d):
    """
    Given two sets of vectors return the l2 distance converted
    to a similarity.
    https://stats.stackexchange.com/q/53068/55807
    :param q: np.array(1, dim)
    :param d: np.array(n_samples, dim)
    """
    if len(q.shape) == 1:
        q = q[np.newaxis, :]
    dists = np.sqrt(np.sum((q-d)**2, axis=1))
    sims = 1/(1+dists[np.newaxis, :])
    return sims


def dot_sim(q, d):
    """
    Given two sets of vectors return the cosine distance.
    :param q: np.array(1, dim)
    :param d: np.array(n_samples, dim)
    """
    if len(q.shape) == 1:
        q = q[np.newaxis, :]
    dot = np.dot(q, d.T)
    return dot


def print_one_pool_nearest_neighbours_pointrec(query_id, qid2queries, ranked_cands, cid2cand_meta, resfile, cand_relevances,
                                               print_only_relevant=False):
    """
    Print the ranked documents for a query.
    :return:
    """
    resfile.write('======================================================================\n')
    resfile.write('query_id: {:s}\n'.format(query_id))
    resfile.write('city: {:s}\n'.format(qid2queries[query_id]['city']))
    resfile.write('country: {:s}\n'.format(qid2queries[query_id]['country']))
    resfile.write('category: {:s}\n'.format(qid2queries[query_id]['category']))
    resfile.write('query: {:s}\n'.format(qid2queries[query_id]['query']))
    resfile.write('===================================\n\n')
    for ranki, ((ndocid, dist), relevance) in enumerate(zip(ranked_cands, cand_relevances)):
        if relevance <= 0 and print_only_relevant:
            continue
        resfile.write('rank: {:d}\n'.format(ranki))
        resfile.write('cand_id: {:s}\n'.format(ndocid))
        resfile.write('rels: {:d}\n'.format(relevance))
        resfile.write('distance: {:.4f}\n'.format(dist))
        resfile.write('city: {:}\n'.format(cid2cand_meta[ndocid]['city']))
        resfile.write('country: {:}\n'.format(cid2cand_meta[ndocid]['country']))
        resfile.write('description: {:s}\n\n'.format(cid2cand_meta[ndocid]['description']))
    resfile.write('======================================================================\n')
    resfile.write('\n')
    

def print_one_pool_nearest_neighbours_sbs16(query_id, qid2queries, ranked_cands, cid2cand_meta, resfile, cand_relevances,
                                            print_only_relevant=False, to_write=200):
    """
    Print the ranked documents for a query.
    :return:
    """
    resfile.write('======================================================================\n')
    resfile.write('query_id: {:s}\n'.format(query_id))
    resfile.write('group: {:s}\n'.format(qid2queries[query_id]['group']))
    resfile.write('title: {:s}\n'.format(qid2queries[query_id]['title']))
    resfile.write('query: {:s}\n'.format(qid2queries[query_id]['query']))
    resfile.write('===================================\n\n')
    for ranki, ((ndocid, dist), relevance) in enumerate(zip(ranked_cands, cand_relevances)):
        if ranki >= to_write:
            break
        if relevance <= 0 and print_only_relevant:
            continue
        resfile.write('rank: {:d}\n'.format(ranki))
        resfile.write('cand_id: {:s}\n'.format(ndocid))
        resfile.write('rels: {:d}\n'.format(relevance))
        resfile.write('distance: {:.4f}\n'.format(dist))
        resfile.write('title: {:}\n'.format(cid2cand_meta[ndocid]['title']))
        resfile.write('author: {:}\n'.format(cid2cand_meta[ndocid]['author']))
        resfile.write('categories: {:}\n'.format(', '.join(cid2cand_meta[ndocid]['categories'])))
        resfile.write('description: {:s}\n\n'.format(cid2cand_meta[ndocid]['description']))
    resfile.write('======================================================================\n')
    resfile.write('\n')
    

def write_trec_eval_out(outfile, qid2ranked_cands, model_name, to_write=200):
    """
    Write out a file with the ranked outputs.
    """
    for qid, ranked_cands in qid2ranked_cands.items():
        for ranki, (candid, dist) in enumerate(ranked_cands):
            if ranki >= to_write:
                break
            outfile.write('{:s}\tQ0\t{:s}\t{:d}\t{:.4f}\t{:s}\n'.format(qid, candid, ranki+1, dist, model_name))


def init_sbmodels(model_name, trained_model_path=None):
    """
    Initialize the model and distance function.
    """
    if model_name in {'sbmpnet1B'}:
        sentbert_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        dist_fun = cosine_sim
    elif model_name in {'sbmpmultiqa'}:
        sentbert_model = SentenceTransformer('sentence-transformers/multi-qa-mpnet-base-cos-v1')
        dist_fun = cosine_sim
    elif model_name in {'sbbertnli'}:
        sentbert_model = SentenceTransformer("sentence-transformers/bert-base-nli-mean-tokens")
        sentbert_model.max_seq_length = 512
        dist_fun = dot_sim
    elif model_name in {'sbcontmsmarco'}:
        sentbert_model = SentenceTransformer("nthakur/contriever-base-msmarco")
        sentbert_model.max_seq_length = 512
        dist_fun = dot_sim
    elif model_name in {'sbcontriver'}:
        word_embedding_model = models.Transformer('facebook/contriever', max_seq_length=512)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode='mean')
        sentbert_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
        dist_fun = dot_sim
    elif model_name in {'sbmpmsmarco'}:
        sentbert_model = SentenceTransformer('sentence-transformers/msmarco-bert-base-dot-v5')
        dist_fun = dot_sim
    elif model_name in {'recinpars'}:
        print(f'Loading: {trained_model_path}')
        with codecs.open(os.path.join(trained_model_path, 'run_info.json'), 'r', 'utf-8') as fp:
            model_hparams = json.load(fp)
            base_model = model_hparams['all_hparams']['base-pt-layer']
        word_embedding_model = models.Transformer(base_model, max_seq_length=512)
        trained_model_fname = os.path.join(trained_model_path, 'model_cur_best.pt')
        word_embedding_model.auto_model.load_state_dict(torch.load(trained_model_fname))
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                       pooling_mode='mean')
        sentbert_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
        dist_fun = l2_sim
    return sentbert_model, dist_fun


def rankwith_sentbert_reps(data_path, run_path, dataset, ranker_name, cand_method=None,
                           trained_ranker_path=None, trained_reranker_path=None, reranker_name='', torerank=200):
    """
    Build pre-trained sentence-bert model representations for pointrec and
    write TREC format outputs.
    :param data_path: string; path from which to read raw documents and queries.
    :param run_path: string; path to save ranked documents to.
    :param dataset: string; {'pointrec'}
    :param trained_ranker_path: string;
    :param ranker_name: string;
    :return:
    """
    start = time.time()
    if dataset == 'pointrec':
        pool_print_func = print_one_pool_nearest_neighbours_pointrec
    elif dataset == 'sbs2016':
        pool_print_func = print_one_pool_nearest_neighbours_sbs16
    queries_fname = os.path.join(data_path, f'{dataset}_queries.json')
    if cand_method:
        pool_fname = os.path.join(data_path, f'test-pid2all_anns-{dataset}-{cand_method}.json')
        cands_fname = os.path.join(data_path, f'candid2all_cands-{dataset}-{cand_method}.json')
        readable_dir = os.path.join(run_path, f'{dataset}-{ranker_name}{reranker_name}-{cand_method}-ranked')
        out_fname = os.path.join(run_path, f'test-pid2pool-{dataset}-{ranker_name}{reranker_name}-{cand_method}-ranked.txt')
    else:
        pool_fname = os.path.join(data_path, f'test-pid2anns-{dataset}.json')
        cands_fname = os.path.join(data_path, f'candid2cands-{dataset}.json')
        readable_dir = os.path.join(run_path, f'{dataset}-{ranker_name}{reranker_name}-ranked')
        out_fname = os.path.join(run_path, f'test-pid2pool-{dataset}-{ranker_name}{reranker_name}-ranked.txt')
    with codecs.open(queries_fname, 'r', 'utf-8') as fp:
        qid2queries = json.load(fp)
        qid2int = collections.OrderedDict([(qid, i) for i, qid in enumerate(list(qid2queries.keys()))])
        queries = [qid2queries[qid]['query'] for qid in qid2int.keys()]
        print(f'Read: {fp.name}')
    uniq_cids = set()
    with codecs.open(pool_fname, 'r', 'utf-8') as fp:
        qid2pool = json.load(fp)
        for qid, pool in qid2pool.items():
            uniq_cids.update(pool['cands'])
        uniq_cids = list(uniq_cids)
        print(f'Read: {fp.name}')
    with codecs.open(cands_fname, 'r', 'utf-8') as fp:
        candid2cands = json.load(fp)
        candid2int = collections.OrderedDict([(cid, i) for i, cid in enumerate(uniq_cids)])
        print(f'Read: {fp.name}')
        cand_docs = [candid2cands[cid]['description'] for cid in candid2int.keys()]
    
    if reranker_name:
        if reranker_name in {'qlft5base', 'qlft5l', 'qlft5xl', 'qlft5xxl'}:
            reranking_model = QueryLikelihoodRR(short_model_name=reranker_name)
        elif reranker_name in {'cemsmlm12'}:
            reranking_model = CrossEncoderRR(short_model_name=reranker_name)
        elif reranker_name in {'recinparsce'}:
            reranking_model = CrossEncoderRR(short_model_name=reranker_name,
                                             trained_model_path=trained_reranker_path)
    sentbert_model, dist_fun = init_sbmodels(ranker_name, trained_ranker_path)
    encoder_pool = sentbert_model.start_multi_process_pool()
    
    # Form query reps.
    print('Encoding queries:')
    query_vectors = sentbert_model.encode_multi_process(queries, encoder_pool, batch_size=64)
    print('Encoded queries: {:}'.format(query_vectors.shape))
    
    # Go over documents and form sb reps for documents.
    print('Encoding documents:')
    document_vectors = sentbert_model.encode_multi_process(cand_docs, encoder_pool, batch_size=64)
    print('Encoded documents: {:}'.format(document_vectors.shape))
    sentbert_model.stop_multi_process_pool(encoder_pool)
    
    # Rank the candidates for every query.
    du.create_dir(readable_dir)
    qid2ranked_cands = dict()
    for qid, pool in qid2pool.items():
        cand_ids = pool['cands']
        cand_rels = pool['relevance_adju']
        cands_idxs = [candid2int[cid] for cid in cand_ids]
        cand2rel = dict(zip(cand_ids, cand_rels))
        cand_reps = document_vectors[cands_idxs, :]
        query_rep = query_vectors[qid2int[qid]]
        sims = dist_fun(query_rep, cand_reps)
        sims = sims[0].tolist()
        assert(len(sims) == len(cand_ids))
        # Sort by first stage ranker sims.
        ranked_cands = sorted(zip(cand_ids, sims), key=lambda t: t[1], reverse=True)
        if reranker_name:
            print(f'Reranking {len(qid2ranked_cands)}: {qid}')
            firststage_ranked_cids = [cid for cid, _ in ranked_cands]
            cand_texts = [candid2cands[cid]['description'] for cid in firststage_ranked_cids[:torerank]]
            cand_scores = reranking_model.reranking_scores(query_text=qid2queries[qid]['query'], cand_texts=cand_texts)
            ranked_cands = sorted(zip(firststage_ranked_cids[:torerank], cand_scores), key=lambda t: t[1], reverse=True)
        qid2ranked_cands[qid] = list(ranked_cands)
        ranked_rels = [cand2rel[cid] for cid, d in ranked_cands]
        # Print the ranked cands per query.
        resfile = codecs.open(os.path.join(readable_dir, f'{qid}-{dataset}-{ranker_name}{reranker_name}-ranked.txt'), 'w', 'utf-8')
        pool_print_func(query_id=qid, qid2queries=qid2queries, ranked_cands=ranked_cands,
                        cid2cand_meta=candid2cands, resfile=resfile, cand_relevances=ranked_rels)
        resfile.close()
    # Write results out for trec_eval to read and eval.
    outf = codecs.open(out_fname, 'w', 'utf-8')
    write_trec_eval_out(outfile=outf, qid2ranked_cands=qid2ranked_cands, model_name=f'{ranker_name}{reranker_name}',
                        to_write=200)
    print(f'Wrote: {outf.name}')
    outf.close()
    print('Took: {:.4f}s'.format(time.time() - start))


def ground_gptwith_sentbert_reps(data_path, run_path, dataset, ranking_model,
                                 cand_method=None, trained_model_path=None):
    """
    Given predictions from GPT3 for narrative queries retrieve the nearest neighbors
    of points of interest from our item corpus with sentence-bert and write TREC format outputs.
    :param data_path: string; path from which to read raw documents and queries.
    :param run_path: string; path to save ranked documents to.
    :param dataset: string; {'pointrec'}
    :param trained_model_path: string;
    :param ranking_model: string;
    :return:
    """
    if dataset == 'pointrec':
        pool_print_func = print_one_pool_nearest_neighbours_pointrec
    elif dataset == 'sbs2016':
        pool_print_func = print_one_pool_nearest_neighbours_sbs16
    llmpreds_fname = os.path.join(run_path, f'narrative_recs-text-davinci-003.json')
    if cand_method:
        pool_fname = os.path.join(data_path, f'test-pid2all_anns-{dataset}-{cand_method}.json')
        cands_fname = os.path.join(data_path, f'candid2all_cands-{dataset}-{cand_method}.json')
        readable_dir = os.path.join(run_path, f'{dataset}-{ranking_model}-{cand_method}-ranked')
    else:
        pool_fname = os.path.join(data_path, f'test-pid2anns-{dataset}.json')
        cands_fname = os.path.join(data_path, f'candid2cands-{dataset}.json')
        readable_dir = os.path.join(run_path, f'{dataset}-{ranking_model}-ranked')
    du.create_dir(readable_dir)
    
    # Read in the candidates.
    with codecs.open(pool_fname, 'r', 'utf-8') as fp:
        qid2pool = json.load(fp)
        print(f'Read: {fp.name}')
    with codecs.open(cands_fname, 'r', 'utf-8') as fp:
        candid2cands = json.load(fp)
        candid2int = collections.OrderedDict([(cid, i) for i, cid in enumerate(list(candid2cands.keys()))])
        print(f'Read: {fp.name}')
        cand_docs = [candid2cands[cid]['description'] for cid in candid2int.keys()]
    
    # Initialize the ranking model.
    sentbert_model, dist_fun = init_sbmodels(ranking_model, trained_model_path)
    encoder_pool = sentbert_model.start_multi_process_pool()
    
    # Go over documents and form sb reps for documents.
    print('Encoding documents:')
    document_vectors = sentbert_model.encode_multi_process(cand_docs, encoder_pool, batch_size=64)
    print('Encoded documents: {:}'.format(document_vectors.shape))
    
    # There are 3 generations per query; ground them all in sequence.
    for llm_run_idx in [0, 1, 2]:
        print(f'LLM Run: {llm_run_idx}')
        if cand_method:
            out_fname = os.path.join(run_path, f'test-pid2pool-{dataset}-hydegpt3-{cand_method}{llm_run_idx}-ranked.txt')
        else:
            out_fname = os.path.join(run_path, f'test-pid2pool-{dataset}-hydegpt3-{llm_run_idx}-ranked.txt')
        with codecs.open(llmpreds_fname, 'r', 'utf-8') as fp:
            qid2generations = json.load(fp)
            qid2int = collections.OrderedDict([(qid, i) for i, qid in enumerate(list(qid2generations.keys()))])
            queries = []
            for qid in sorted(qid2generations):
                generations_d = qid2generations[qid]['llm_responses']
                text_list = generations_d[llm_run_idx]['choices'][0]['text']
                # Generations are a numberd list.
                pred_list = []
                for line in text_list.strip().split('\n'):
                    if re.match('^[0-9]+', line.strip()):
                        pred_list.append(re.sub('[0-9]+.', '', line).strip())
                if len(pred_list) != 10:  # There should be 10 generations but generations may be incorrect
                    print(f'Over/under generated: {qid}; {len(pred_list)}')
                queries.append(pred_list)
            print(f'Read: {fp.name}')
        
        # Form "query" reps where the items are the top-10 LLM predictions.
        all_query_vectors = []
        for userq_items in queries:
            userq_vectors = sentbert_model.encode_multi_process(userq_items, encoder_pool)
            all_query_vectors.append(userq_vectors)
        print('Encoded queries: {:}'.format(len(all_query_vectors)))
        
        # Rank the candidates for every query.
        qid2ranked_cands = dict()
        for qid in qid2int:
            pool = qid2pool[qid]
            cands_idxs = [candid2int[cid] for cid in pool['cands']]
            cand2rel = dict(zip(pool['cands'], pool['relevance_adju']))
            cand_reps = document_vectors[cands_idxs, :]
            # Get all the predicted items embeddings.
            query_reps = all_query_vectors[qid2int[qid]]
            sims = dist_fun(query_reps, cand_reps)
            # Go over each predicted item and get the min distance
            cand2allsims = collections.defaultdict(list)
            for i in range(sims.shape[0]):
                item_sims = sims[i]
                for cand, sim in zip(pool['cands'], item_sims):
                    cand2allsims[cand].append(sim)
            cand2sim = [(c, max(allsims)) for c, allsims in cand2allsims.items()]
            assert (len(cand2sim) == len(pool['cands']))
            ranked_cands = sorted(cand2sim, key=lambda t: t[1], reverse=True)
            qid2ranked_cands[qid] = list(ranked_cands)
            ranked_rels = [cand2rel[cid] for cid, d in ranked_cands]
            if llm_run_idx == 0:  # Only print readable outputs once.
                # Print the ranked cands per query.
                resfile = codecs.open(os.path.join(readable_dir, f'{qid}-{dataset}-{ranking_model}-ranked.txt'), 'w', 'utf-8')
                pool_print_func(query_id=qid, qid2queries=qid2generations, ranked_cands=ranked_cands,
                                cid2cand_meta=candid2cands, resfile=resfile, cand_relevances=ranked_rels)
                resfile.close()
        # Write results out for trec_eval to read and eval.
        outf = codecs.open(out_fname, 'w', 'utf-8')
        write_trec_eval_out(outfile=outf, qid2ranked_cands=qid2ranked_cands, model_name=ranking_model, to_write=200)
        print(f'Wrote: {outf.name}\n')
        outf.close()
    sentbert_model.stop_multi_process_pool(encoder_pool)
    

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='subcommand',
                                       help='The action to perform.')
    # Get tfidf reps.
    build_vecs_args = subparsers.add_parser('rank_cands')
    build_vecs_args.add_argument('--ranker_name', required=True,
                                 choices=['sbmpnet1B', 'sbmpmultiqa', 'sbmpmsmarco', 'okapibm25',
                                          'sbbertnli', 'sbcontmsmarco', 'sbcontriver', 'recinpars'],
                                 help='The name of the model to run.')
    build_vecs_args.add_argument('--reranker_name', required=False, default=None,
                                 choices=['qlft5base', 'qlft5xl', 'cemsmlm12', 'recinparsce'],
                                 help='The name of the model to use for reranking.')
    build_vecs_args.add_argument('--torerank', required=False, type=int,
                                 choices=[100, 200],
                                 help='The number of first stage candidates to rerank.')
    build_vecs_args.add_argument('--cand_method',
                                 # 'tfidf', 'mpnet1b' - not allowing these cand methods for now. kind of not clean.
                                 # they are used for sbs16 though - its as clean as that data gets.
                                 choices=['city', 'citycat', 'mpnet1b'],
                                 help='The candidate set to use.')
    build_vecs_args.add_argument('--dataset', required=True,
                                 choices=['pointrec', 'sbs2016'],
                                 help='The dataset to predict on.')
    build_vecs_args.add_argument('--data_path', required=True,
                                 help='Path to directory with json data.')
    build_vecs_args.add_argument('--run_path', required=True,
                                 help='Path to directory to save all run items to.')
    build_vecs_args.add_argument('--ranker_model_path',
                                 help='Path to directory with trained model to '
                                      'use for getting reps.')
    build_vecs_args.add_argument('--reranker_model_path', default=None,
                                 help='Path to directory with trained model to '
                                      'use for getting reps.')
    build_vecs_args.add_argument('--run_name',
                                 help='Basename for the trained model directory.')
    # Get tfidf reps.
    ground_args = subparsers.add_parser('ground_in_cands')
    ground_args.add_argument('--ranking_model', required=True,
                             choices=['sbmpnet1B', 'sbmpmultiqa', 'sbmpmsmarco', 'okapibm25',
                                      'sbbertnli', 'sbcontmsmarco', 'sbcontriver', 'recinpars'],
                             help='The name of the model to run.')
    ground_args.add_argument('--cand_method',  # 'mpnet1b' is mpnet1b run on the query meta and descr in sbs16.
                             choices=['city', 'citycat', 'mpnet1b'],
                             help='The candidate set to use.')
    ground_args.add_argument('--dataset', required=True,
                             choices=['pointrec', 'sbs2016'],
                             help='The dataset to predict on.')
    ground_args.add_argument('--data_path', required=True,
                             help='Path to directory with json data.')
    ground_args.add_argument('--run_path', required=True,
                             help='Path to directory to save all run items to.')
    ground_args.add_argument('--model_path',
                             help='Path to directory with trained model to '
                                  'use for getting reps.')
    ground_args.add_argument('--run_name',
                             help='Basename for the trained model directory.')
    cl_args = parser.parse_args()
    if cl_args.subcommand == 'rank_cands':
        # if cl_args.run_name and cl_args.reranker_name == None:
        #     run_path = os.path.join(cl_args.run_path, cl_args.run_name)
        # else:
        run_path = cl_args.run_path
        if cl_args.ranker_name == 'okapibm25':
            if cl_args.reranker_name:
                rankwith_bm25(data_path=cl_args.data_path, run_path=run_path,
                              dataset=cl_args.dataset, ranker_name=cl_args.ranker_name, cand_method=cl_args.cand_method,
                              reranker_name=cl_args.reranker_name, torerank=cl_args.torerank,
                              trained_reranker_path=cl_args.reranker_model_path)
            else:
                rankwith_bm25(data_path=cl_args.data_path, run_path=run_path,
                              dataset=cl_args.dataset, ranker_name=cl_args.ranker_name,
                              cand_method=cl_args.cand_method)
        elif cl_args.ranker_name in {'sbmpnet1B', 'sbmpmultiqa', 'sbmpmsmarco', 'sbbertnli', 'sbcontmsmarco',
                                     'sbcontriver'}:
            if cl_args.reranker_name:
                rankwith_sentbert_reps(data_path=cl_args.data_path, run_path=run_path,
                                       dataset=cl_args.dataset, cand_method=cl_args.cand_method,
                                       ranker_name=cl_args.ranker_name,
                                       reranker_name=cl_args.reranker_name, torerank=cl_args.torerank,
                                       trained_reranker_path=cl_args.reranker_model_path)
            else:
                rankwith_sentbert_reps(data_path=cl_args.data_path, run_path=run_path,
                                       dataset=cl_args.dataset, ranker_name=cl_args.ranker_name,
                                       cand_method=cl_args.cand_method)
        elif cl_args.ranker_name in {'recinpars'}:
            if cl_args.reranker_name:
                rankwith_sentbert_reps(data_path=cl_args.data_path, run_path=run_path,
                                       dataset=cl_args.dataset, cand_method=cl_args.cand_method,
                                       trained_ranker_path=cl_args.ranker_model_path, ranker_name=cl_args.ranker_name,
                                       trained_reranker_path=cl_args.reranker_model_path,
                                       reranker_name=cl_args.reranker_name, torerank=cl_args.torerank)
            else:
                rankwith_sentbert_reps(data_path=cl_args.data_path, run_path=run_path,
                                       dataset=cl_args.dataset, ranker_name=cl_args.ranker_name,
                                       trained_ranker_path=cl_args.ranker_model_path, cand_method=cl_args.cand_method)
    elif cl_args.subcommand == 'ground_in_cands':
        ground_gptwith_sentbert_reps(data_path=cl_args.data_path, run_path=cl_args.run_path,
                                     dataset=cl_args.dataset, ranking_model=cl_args.ranking_model,
                                     cand_method=cl_args.cand_method, trained_model_path=cl_args.model_path)


if __name__ == '__main__':
    main()
