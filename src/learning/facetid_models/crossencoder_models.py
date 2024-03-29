"""
- Use the Sentence-Transformer implementation and modify the
CrossEncoder to suit your needs.
"""
import logging
import os
import random
from typing import Dict, Type, Callable, List
from tqdm.autonotebook import tqdm, trange
import transformers
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.evaluation import SentenceEvaluator


class MyCrossEncoder(CrossEncoder):
    """
    Allow better logging of the training with comet.
    """
    def fit(self,
            cml_experiment,
            train_dataloader: DataLoader,
            evaluator: SentenceEvaluator = None,
            epochs: int = 1,
            loss_fct = None,
            activation_fct = nn.Identity(),
            scheduler: str = 'WarmupLinear',
            warmup_steps: int = 10000,
            optimizer_class: Type[Optimizer] = transformers.AdamW,
            optimizer_params: Dict[str, object] = {'lr': 2e-5},
            weight_decay: float = 0.01,
            evaluation_steps: int = 0,
            output_path: str = None,
            save_best_model: bool = True,
            max_grad_norm: float = 1,
            use_amp: bool = False,
            callback: Callable[[float, int, int], None] = None,
            ):
        """
        Train the model with the given training objective
        Each training objective is sampled in turn for one batch.
        We sample only as many batches from each objective as there are in the smallest one
        to make sure of equal training with each dataset.
        
        :param cml_experiment: comet ml Experiment object for tracking training.
        :param train_dataloader: DataLoader with training InputExamples
        :param evaluator: An evaluator (sentence_transformers.evaluation) evaluates the model performance during training on held-out dev data. It is used to determine the best model that is saved to disc.
        :param epochs: Number of epochs for training
        :param loss_fct: Which loss function to use for training. If None, will use nn.BCEWithLogitsLoss() if self.config.num_labels == 1 else nn.CrossEntropyLoss()
        :param activation_fct: Activation function applied on top of logits output of model.
        :param scheduler: Learning rate scheduler. Available schedulers: constantlr, warmupconstant, warmuplinear, warmupcosine, warmupcosinewithhardrestarts
        :param warmup_steps: Behavior depends on the scheduler. For WarmupLinear (default), the learning rate is increased from o up to the maximal learning rate. After these many training steps, the learning rate is decreased linearly back to zero.
        :param optimizer_class: Optimizer
        :param optimizer_params: Optimizer parameters
        :param weight_decay: Weight decay for model parameters
        :param evaluation_steps: If > 0, evaluate the model using evaluator after each number of training steps
        :param output_path: Storage path for the model and evaluation files
        :param save_best_model: If true, the best model (according to evaluator) is stored at output_path
        :param max_grad_norm: Used for gradient normalization.
        :param use_amp: Use Automatic Mixed Precision (AMP). Only for Pytorch >= 1.6.0
        :param callback: Callback function that is invoked after each evaluation.
                It must accept the following three parameters in this order:
                `score`, `epoch`, `steps`
        """
        train_dataloader.collate_fn = self.smart_batching_collate

        if use_amp:
            from torch.cuda.amp import autocast
            scaler = torch.cuda.amp.GradScaler()

        self.model.to(self._target_device)

        if output_path is not None:
            os.makedirs(output_path, exist_ok=True)

        self.best_score = -9999999
        num_train_steps = int(len(train_dataloader) * epochs)

        # Prepare optimizers
        param_optimizer = list(self.model.named_parameters())

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)

        if isinstance(scheduler, str):
            scheduler = SentenceTransformer._get_scheduler(optimizer, scheduler=scheduler, warmup_steps=warmup_steps, t_total=num_train_steps)

        if loss_fct is None:
            loss_fct = nn.BCEWithLogitsLoss() if self.config.num_labels == 1 else nn.CrossEntropyLoss()

        skip_scheduler = False
        training_steps = 0
        # for epoch in trange(epochs, desc="Epoch"):
        for epoch in range(epochs):
            self.model.zero_grad()
            self.model.train()
            # for features, labels in tqdm(train_dataloader, desc="Iteration", smoothing=0.05):
            for features, labels in train_dataloader:
                if use_amp:
                    with autocast():
                        model_predictions = self.model(**features, return_dict=True)
                        logits = activation_fct(model_predictions.logits)
                        if self.config.num_labels == 1:
                            logits = logits.view(-1)
                        loss_value = loss_fct(logits, labels)

                    scale_before_step = scaler.get_scale()
                    scaler.scale(loss_value).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()

                    skip_scheduler = scaler.get_scale() != scale_before_step
                else:
                    model_predictions = self.model(**features, return_dict=True)
                    logits = activation_fct(model_predictions.logits)
                    if self.config.num_labels == 1:
                        logits = logits.view(-1)
                    loss_value = loss_fct(logits, labels)
                    loss_value.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    optimizer.step()
                optimizer.zero_grad()
                
                if not skip_scheduler:
                    scheduler.step()

                training_steps += 1
                
                cml_experiment.log_metric("train-loss", loss_value, step=training_steps, epoch=epoch)
                logging.info('Epoch: {:d}; Iteration: {:d}/{:d}; Loss: {:}'.format(epoch, training_steps,
                                                                                   num_train_steps, loss_value))
                if evaluator is not None and evaluation_steps > 0 and training_steps % evaluation_steps == 0:
                    dev_score = self._eval_during_training(evaluator, output_path, save_best_model, epoch,
                                                           training_steps, callback)
                    cml_experiment.log_metric("dev-mrr", dev_score, step=training_steps)
                    self.model.zero_grad()
                    self.model.train()

            if evaluator is not None:
                self._eval_during_training(evaluator, output_path, save_best_model, epoch, -1, callback)

    def _eval_during_training(self, evaluator, output_path, save_best_model, epoch, steps, callback):
        """Runs evaluation during the training"""
        if evaluator is not None:
            score = evaluator(self, output_path=output_path, epoch=epoch, steps=steps)
            if callback is not None:
                callback(score, epoch, steps)
            if score > self.best_score:
                self.best_score = score
                if save_best_model:
                    self.save(output_path)
            return score


class SoftMaxCrossEncoder(MyCrossEncoder):
    """
    Train the model with
    """
    def fit(self,
            cml_experiment,
            positive_examples,
            negative_examples,
            batch_size,
            neg_per_pos,
            evaluator: SentenceEvaluator = None,
            epochs: int = 1,
            loss_fct=None,
            activation_fct=nn.Identity(),
            scheduler: str = 'WarmupLinear',
            warmup_steps: int = 10000,
            optimizer_class: Type[Optimizer] = transformers.AdamW,
            optimizer_params: Dict[str, object] = {'lr': 2e-5},
            weight_decay: float = 0.01,
            evaluation_steps: int = 0,
            output_path: str = None,
            save_best_model: bool = True,
            max_grad_norm: float = 1,
            use_amp: bool = False,
            callback: Callable[[float, int, int], None] = None,
            ):
        """
        Train the model with the given training objective
        Each training objective is sampled in turn for one batch.
        We sample only as many batches from each objective as there are in the smallest one
        to make sure of equal training with each dataset.

        :param cml_experiment: comet ml Experiment object for tracking training.
        :param positive_examples: list(InputExamples)
        :param negative_examples: list(list(InputExamples))
        :param batch_size: int
        :param neg_per_pos: int
        :param evaluator: An evaluator (sentence_transformers.evaluation) evaluates the model performance during training on held-out dev data. It is used to determine the best model that is saved to disc.
        :param epochs: Number of epochs for training
        :param loss_fct: Which loss function to use for training. If None, will use nn.BCEWithLogitsLoss() if self.config.num_labels == 1 else nn.CrossEntropyLoss()
        :param activation_fct: Activation function applied on top of logits output of model.
        :param scheduler: Learning rate scheduler. Available schedulers: constantlr, warmupconstant, warmuplinear, warmupcosine, warmupcosinewithhardrestarts
        :param warmup_steps: Behavior depends on the scheduler. For WarmupLinear (default), the learning rate is increased from o up to the maximal learning rate. After these many training steps, the learning rate is decreased linearly back to zero.
        :param optimizer_class: Optimizer
        :param optimizer_params: Optimizer parameters
        :param weight_decay: Weight decay for model parameters
        :param evaluation_steps: If > 0, evaluate the model using evaluator after each number of training steps
        :param output_path: Storage path for the model and evaluation files
        :param save_best_model: If true, the best model (according to evaluator) is stored at output_path
        :param max_grad_norm: Used for gradient normalization.
        :param use_amp: Use Automatic Mixed Precision (AMP). Only for Pytorch >= 1.6.0
        :param callback: Callback function that is invoked after each evaluation.
                It must accept the following three parameters in this order:
                `score`, `epoch`, `steps`
        """
        self.best_score = -9999999
        num_examples = len(positive_examples)
        num_batches = num_examples // batch_size
        unbatchable_examples = num_examples % batch_size
        num_train_steps = int(num_batches * epochs)
        if use_amp:
            from torch.cuda.amp import autocast
            scaler = torch.cuda.amp.GradScaler()
        
        self.model.to(self._target_device)
        
        if output_path is not None:
            os.makedirs(output_path, exist_ok=True)
        
        # Prepare optimizers
        param_optimizer = list(self.model.named_parameters())
        
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        
        optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)
        
        if isinstance(scheduler, str):
            scheduler = SentenceTransformer._get_scheduler(optimizer, scheduler=scheduler, warmup_steps=warmup_steps,
                                                           t_total=num_train_steps)
        
        loss_fct = nn.CrossEntropyLoss(reduction='mean')
        
        skip_scheduler = False
        training_steps = 0
        for epoch in range(epochs):
            # Shuffle the examples at the start of the epoch; Discard the examples which cant
            # fit exactly in a batch of size batch_size
            shuf_idxs = list(range(num_examples-unbatchable_examples))
            random.shuffle(shuf_idxs)
            shuf_pos_examples = []  # list(InputExample)
            shuf_neg_examples = []  # list(InputExample)
            for i in shuf_idxs:
                shuf_pos_examples.append(positive_examples[i])
                shuf_neg_examples.extend(negative_examples[i])
            pos_ex_dataloader = DataLoader(shuf_pos_examples, shuffle=False, batch_size=batch_size)
            neg_ex_dataloader = DataLoader(shuf_neg_examples, shuffle=False, batch_size=batch_size*neg_per_pos)
            pos_ex_dataloader.collate_fn = self.smart_batching_collate
            neg_ex_dataloader.collate_fn = self.smart_batching_collate
            # Then go over the batches.
            self.model.zero_grad()
            self.model.train()
            for (pos_features, _), (neg_features, _) in zip(pos_ex_dataloader, neg_ex_dataloader):
                labels = torch.zeros(batch_size, dtype=torch.long)
                labels = labels.to(self._target_device)
                if use_amp:
                    with autocast():
                        pos_model_outputs = self.model(**pos_features, return_dict=True)
                        pos_logits = activation_fct(pos_model_outputs.logits)
                        pos_logits = pos_logits.view(batch_size, 1)
                        neg_model_outputs = self.model(**neg_features, return_dict=True)
                        neg_logits = activation_fct(neg_model_outputs.logits)
                        neg_logits = neg_logits.view(batch_size, neg_per_pos)
                        all_logits = torch.hstack([pos_logits, neg_logits])
                        loss_value = loss_fct(all_logits, labels)
                    
                    scale_before_step = scaler.get_scale()
                    scaler.scale(loss_value).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    
                    skip_scheduler = scaler.get_scale() != scale_before_step
                else:
                    pos_model_outputs = self.model(**pos_features, return_dict=True)
                    pos_logits = activation_fct(pos_model_outputs.logits)
                    pos_logits = pos_logits.view(batch_size, 1)
                    neg_model_outputs = self.model(**neg_features, return_dict=True)
                    neg_logits = activation_fct(neg_model_outputs.logits)
                    neg_logits = neg_logits.view(batch_size, neg_per_pos)
                    all_logits = torch.hstack([pos_logits, neg_logits])
                    loss_value = loss_fct(all_logits, labels)
                    loss_value.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    optimizer.step()
                optimizer.zero_grad()
                
                if not skip_scheduler:
                    scheduler.step()
                
                training_steps += 1
                
                cml_experiment.log_metric("train-loss", loss_value, step=training_steps, epoch=epoch)
                logging.info('Epoch: {:d}; Iteration: {:d}/{:d}; Loss: {:}'.format(epoch, training_steps,
                                                                                   num_train_steps, loss_value))
                if evaluator is not None and evaluation_steps > 0 and training_steps % evaluation_steps == 0:
                    dev_score = self._eval_during_training(evaluator, output_path, save_best_model, epoch,
                                                           training_steps, callback)
                    cml_experiment.log_metric("dev-mrr", dev_score, step=training_steps)
                    self.model.zero_grad()
                    self.model.train()
            
            if evaluator is not None:
                dev_score = self._eval_during_training(evaluator, output_path, save_best_model, epoch, -1, callback)
                cml_experiment.log_metric("dev-mrr", dev_score, step=training_steps)
