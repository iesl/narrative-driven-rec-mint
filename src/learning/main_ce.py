"""
Train a cross-encoder using the Sentence-Transformers library.
- Treat the task as a classification task, i.e a pointwise loss.
- Read pre-generated positives and negatives.
"""
import pprint
import random
import sys
import os
import logging
import comet_ml as cml
import re
import time
import codecs, json
import argparse
import collections
from torch.utils.data import DataLoader
from torch import nn
from sentence_transformers import InputExample
from sentence_transformers.cross_encoder.evaluation import CERerankingEvaluator
from .facetid_models import crossencoder_models


def pointwise_train_data(data_path, all_hparams):
    """
    Read the training data for pointwise training.
    The file contains a series of positive and negative training query-doc pairs.
    """
    with codecs.open(os.path.join(data_path, f'ce-train-{all_hparams["train_suffix"]}.jsonl'), 'r', 'utf-8') as fp:
        train_samples = []
        num_train = 0
        for line in fp:
            if num_train > all_hparams['train_size']:
                break
            ex_dict = json.loads(line.strip())
            train_samples.append(InputExample(texts=[ex_dict['query_text'], ex_dict['doc_text']],
                                              label=ex_dict['label']))
            num_train += 1
    logging.info(f'Read train examples: {len(train_samples)}')
    return train_samples


def listwise_train_data(data_path, all_hparams):
    """
    Read the training data for pointwise training and create listwise samples from it.
    - For every positive sample get neg_per_pos negatives.
    -- If there are fewer negatives than positive*neg_per_pos
        then resample negatives till you get the needed number.
    - The model will optimize softmax over the positives and negatives.
    """
    neg_per_pos = all_hparams['neg_per_pos']
    uid2pos = collections.defaultdict(list)
    uid2neg = collections.defaultdict(list)
    docid2doc = dict()
    with codecs.open(os.path.join(data_path, f'ce-train-{all_hparams["train_suffix"]}.jsonl'), 'r', 'utf-8') as fp:
        for line in fp:
            ex_dict = json.loads(line.strip())
            docid2doc[ex_dict['cited_pids'][0]] = ex_dict['query_text']
            docid2doc[ex_dict['cited_pids'][1]] = ex_dict['doc_text']
            if ex_dict['label'] == 1:
                uid2pos[ex_dict['user_id']].append(ex_dict['cited_pids'][1])
            elif ex_dict['label'] == 0:
                uid2neg[ex_dict['user_id']].append(ex_dict['cited_pids'][1])
        assert (len(uid2pos) == len(uid2neg))
        # Train size is the number of users.
        train_uids = list(uid2pos.keys())[:all_hparams['train_users']]
        uid2pos = {uid: uid2pos[uid] for uid in train_uids}
        uid2neg = {uid: uid2neg[uid] for uid in train_uids}
        logging.info('Train users: {:}'.format(len(uid2pos)))
    pos_examples = []  # list(InputExamples)
    neg_examples = []  # list(list(InputExamples))
    available_le_needed = 0
    for uid, pos_docids in uid2pos.items():
        pos_ex_count = len(pos_docids)
        user_pos_exs = []
        for pos_id in pos_docids:
            user_pos_exs.append(InputExample(texts=[docid2doc[uid], docid2doc[pos_id]], label=1))
        pos_examples.extend(user_pos_exs)
        # If there are fewer negatives than pos_exs*neg_per_pos then re-sample
        # negatives and add them to the negatives. This happens rarely.
        neg_docids = uid2neg[uid]
        available_negs = len(neg_docids)
        needed_negs = pos_ex_count*neg_per_pos
        if available_negs < needed_negs:
            to_sample = needed_negs - available_negs
            resampled_negs = []
            for i in range(to_sample):
                resampled_negs.append(random.choice(neg_docids))
            neg_docids.extend(resampled_negs)
            available_le_needed += 1
        user_neg_exs = []
        for ni in range(pos_ex_count):
            per_pos_negs = []
            for neg_id in neg_docids[ni*neg_per_pos:(ni+1)*neg_per_pos]:
                per_pos_negs.append(InputExample(texts=[docid2doc[uid], docid2doc[neg_id]], label=0))
            user_neg_exs.append(per_pos_negs)
        neg_examples.extend(user_neg_exs)
    logging.info('Users with available negatives < needed negatives: {:}'.format(available_le_needed))
    logging.info('Positive examples: {:}; Negatives per positive: {:}'.
                 format(len(pos_examples), neg_per_pos))
    assert len(pos_examples) == len(neg_examples)
    return pos_examples, neg_examples


def train_model(model_name, data_path, config_path, run_path, cl_args):
    """
    - Load training data.
    - Load
    """
    run_name = os.path.basename(run_path)
    # Load label maps and configs.
    with codecs.open(config_path, 'r', 'utf-8') as fp:
        all_hparams = json.load(fp)
    
    cml_experiment = cml.Experiment(project_name='2021-edit-expertise', display_summary_level=0)
    cml_experiment.log_parameters(all_hparams)
    cml_experiment.set_name(run_name)
    # Save the name of the screen session the experiment is running in.
    cml_experiment.add_tags([cl_args.dataset, cl_args.model_name, os.environ['STY']])
    
    # Unpack hyperparameter settings.
    logging.info('All hyperparams:')
    logging.info(pprint.pformat(all_hparams))
    
    # Save hyperparams to disk.
    run_info = {'all_hparams': all_hparams}
    with codecs.open(os.path.join(run_path, 'run_info.json'), 'w', 'utf-8') as fp:
        json.dump(run_info, fp)
        
    # Read the dev data.
    with codecs.open(os.path.join(data_path, f'ce-dev-{all_hparams["train_suffix"]}.jsonl'), 'r', 'utf-8') as fp:
        dev_samples = {}
        num_dev = 0
        for line in fp:
            if num_dev > all_hparams['dev_size']:
                break
            ex_dict = json.loads(line.strip())
            pos_docs = set([d['doc_text'] for d in ex_dict['positive_docs']][:5])
            neg_docs = set([d['doc_text'] for d in ex_dict['negative_docs']][:200])
            dev_samples[ex_dict['user_id']] = {'query': ex_dict['query_text'],
                                               'positive': pos_docs,
                                               'negative': neg_docs}
            num_dev += 1
        
        logging.info(f'Read dev examples: {len(dev_samples)}')
        evaluator = CERerankingEvaluator(dev_samples, name='train-eval', mrr_at_k=50)
    
    # Read the training data.
    train_criterion = all_hparams.get('criterion', 'BCEWithLogitsLoss')
    if train_criterion == 'BCEWithLogitsLoss':
        train_samples = pointwise_train_data(data_path=data_path, all_hparams=all_hparams)
        train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=all_hparams['batch_size'])
    elif train_criterion == 'CrossEntropyLoss':
        pos_samples, neg_samples = listwise_train_data(data_path=data_path, all_hparams=all_hparams)
        
    # Initialize model.
    start = time.time()
    if train_criterion == 'BCEWithLogitsLoss':
        ce_model = crossencoder_models.MyCrossEncoder(model_name=all_hparams["base-pt-layer"],
                                                      num_labels=1, max_length=512)
    elif train_criterion == 'CrossEntropyLoss':
        ce_model = crossencoder_models.SoftMaxCrossEncoder(model_name=all_hparams["base-pt-layer"],
                                                           num_labels=1, max_length=512,
                                                           default_activation_function=nn.Identity())
    
    logging.info(ce_model.model)
    
    # Initialize the trainer.
    if train_criterion == 'BCEWithLogitsLoss':
        ce_model.fit(cml_experiment=cml_experiment,
                     train_dataloader=train_dataloader,
                     evaluator=evaluator,
                     epochs=all_hparams['num_epochs'],
                     optimizer_params={'lr': all_hparams['learning_rate']},
                     evaluation_steps=all_hparams['es_check_every'],
                     warmup_steps=all_hparams['num_warmup_steps'],
                     output_path=run_path,
                     use_amp=True)
    elif train_criterion == 'CrossEntropyLoss':
        ce_model.fit(cml_experiment=cml_experiment,
                     positive_examples=pos_samples, negative_examples=neg_samples,
                     batch_size=all_hparams['batch_size'], neg_per_pos=all_hparams['neg_per_pos'],
                     evaluator=evaluator,
                     epochs=all_hparams['num_epochs'],
                     optimizer_params={'lr': all_hparams['learning_rate']},
                     evaluation_steps=all_hparams['es_check_every'],
                     warmup_steps=all_hparams['num_warmup_steps'],
                     output_path=run_path,
                     use_amp=True)
    
    # Save latest model
    ce_model.save(run_path + '-final')
    logging.info('Took: {:.2f}s'.format(time.time()-start))
    
    
def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='subcommand',
                                       help='The action to perform.')
    # Train the model.
    train_args = subparsers.add_parser('train_model')
    # Where to get what.
    train_args.add_argument('--model_name', required=True,
                            choices=['recinparsce'],
                            help='The name of the model to train.')
    train_args.add_argument('--dataset', required=True,
                            choices=['yelppoi'],
                            help='The dataset to train and predict on.')
    train_args.add_argument('--num_gpus', required=True, type=int,
                            help='Number of GPUs to train on/number of processes running parallel training.')
    train_args.add_argument('--data_path', required=True,
                            help='Path to the jsonl dataset.')
    train_args.add_argument('--run_path', required=True,
                            help='Path to directory to save all run items to.')
    train_args.add_argument('--config_path', required=True,
                            help='Path to directory json config file for model.')
    cl_args = parser.parse_args()
    # If a log file was passed then write to it.

    try:
        logging.basicConfig(level='INFO', format='%(message)s',
                            filename=cl_args.log_fname)
        # Print the called script and its args to the log.
        logging.info(' '.join(sys.argv))
    # Else just write to stdout.
    except AttributeError:
        logging.basicConfig(level='INFO', format='%(message)s',
                            stream=sys.stdout)
        # Print the called script and its args to the log.
        logging.info(' '.join(sys.argv))

    if cl_args.subcommand == 'train_model':
        train_model(model_name=cl_args.model_name, data_path=cl_args.data_path,
                    run_path=cl_args.run_path, config_path=cl_args.config_path, cl_args=cl_args)


if __name__ == '__main__':
    main()
