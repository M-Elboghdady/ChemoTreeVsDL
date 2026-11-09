import torch
import numpy as np
import os
import psutil
import pandas as pd
import pickle
import time
from tqdm import tqdm
from torch.optim import AdamW
from ts_model_training.evaluator import EvaluatorPretrain, EvaluatorTrain
from ts_model_training.saver import ResultSaverGrid, ResultSaverBest
from ts_model_training.tracker import Tracker
from ts_model_training.utils import format_dict, count_parameters
from math import ceil

class Trainer:
    def __init__(self, args, model, batcher):
        # get data and params
        self.args = args
        self.model = model
        self.batcher = batcher
        self.splits = batcher.splits
        self.test_bool = len(self.splits["test"]) != 0 # there is a test set (supervised)
        self.val_bool = len(self.splits["val"]) != 0 # there is a validation set (train w patience)
        if not self.val_bool:
            self.args.max_epochs = int(ceil(self.args.mean_epochs)) - self.args.patience
            self.args.logger.write(f"Max epochs reset to: {self.args.max_epochs}")   
        self._setup_output()
        self._initialise_model()
        self.optimizer = self._setup_optimizer()
        self.evaluator = self._setup_evaluator()
        self.saver = self._setup_saver()
        self.tracker = Tracker(self.args)

        # early stopping / training state
        self.wait = args.patience
        self.patience_reached = False
        self.best_val_metric = -np.inf
        self.best_val_res = None
        self.best_test_res = None
        self.step_total = 0
        self.train_losses_per_epoch = []
        self.val_losses_per_epoch = []            

        # calculate batches per epoch and steps
        self.num_batches_per_epoch = int(np.ceil(len(self.splits["train"]) / self.args.train_batch_size))
        
        # TO DO : CHECK BEST batch num calculation
        #num_neg = (self.batcher.y[self.batcher.splits['train']] == 0).sum()
        #num_neg_batch = self.args.train_batch_size - self.args.stratify_batch
        #self.num_batches_per_epoch =int(np.ceil(num_neg/num_neg_batch))
        
        self.args.logger.write(f'\nNo. of training batches per epoch = {self.num_batches_per_epoch}')
        
    def _setup_optimizer(self):
        return AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=float(self.args.lr), 
            weight_decay=float(self.args.weight_decay),
        )
        
    def _setup_evaluator(self):
        if self.args.train_mode == "pretrain":
            return EvaluatorPretrain(self.args, self.batcher)
        return EvaluatorTrain(self.args, self.batcher)
    
    def _setup_output(self):       
        self.output_dir = self.args.paths["output_path"]
        self.model_path_best = os.path.join(self.output_dir, 'checkpoint_best.bin')
        self.args.logger.write(f"\nSaving results in folder: {self.output_dir}")
        
    def _setup_saver(self):
        if self.args.cv_mode == "grid": #self.args.grid == "nested" and 
            return ResultSaverGrid(self.output_dir)
        return ResultSaverBest(self.output_dir)
        
    def _initialise_model(self):
        self.args.logger.write(f"\nSelected model {self.args.model_type} using device {self.args.device} in {self.args.train_mode} mode")
        # if pretrained model available, copy parameters
        if self.args.train_mode == "finetune":
            pt_state_dict = torch.load(self.args.pt_dict_path, map_location=self.args.device)
            missing_keys, unexpected_keys = self.model.load_state_dict(pt_state_dict, strict=False)
            if missing_keys:
                self.args.logger.write(f"Warning: Missing keys in loaded state dict: {missing_keys}")
            if unexpected_keys:
                self.args.logger.write(f"Warning: Unexpected keys in loaded state dict: {unexpected_keys}")

            if self.args.freeze:
                # freeze all parameters
                for param in self.model.parameters():
                    param.requires_grad = False
                # unfreeze only desired layers
                for param in self.model.fusion_att.parameters():
                    param.requires_grad = True
                for param in self.model.forecast_head.parameters():
                    param.requires_grad = True
                for param in self.model.binary_head.parameters():
                    param.requires_grad = True
            # check trainable parameters     
            self.args.logger.write("Trainable parameters : ")
            for name, param in self.model.named_parameters():
                self.args.logger.write(f"{name}: {param.requires_grad}")
        # count model parameters
        self.param_info = count_parameters(self.args.logger, self.model)


    def train(self):
        self.evaluate_at_time(-1)
        self.model.train()

        for epoch in range(1, self.args.max_epochs + 1):
            self.args.logger.write(f"\n########### EPOCH {epoch} ############")
            self.tracker.reset_train_epoch()
            self.train_one_epoch()
            self.tracker.update_train_epoch()
            self.evaluate_at_epoch(self.step_total)
            if self.patience_reached:
                break
            
        # final logging and saving
        self.collect_final_results()
        self.saver.save(self.final_results)


    def collect_final_results(self):
        
        # if model not trained with patience - save last model
        if not self.val_bool:
            self.args.logger.write('\nSaving ckpt at ' + self.model_path_best)
            torch.save(self.model.state_dict(), self.model_path_best)

        # apply inference on test set using best model
        if self.test_bool:
            test_batches = self.evaluator.prepare_batches("test")
            self.tracker.compute_inference_metrics(self.best_model, test_batches) 
        
        losses = {
            "epoch": list(range(1, len(self.train_losses_per_epoch) + 1)),
            "epoch_time": self.tracker.epoch_train_times,
            "epoch_memory": self.tracker.epoch_train_memory,
            "train_loss": self.train_losses_per_epoch,
        }
        if len(self.val_losses_per_epoch) > 0:
            losses["val_loss"] = self.val_losses_per_epoch
        
        self.final_results = {
            "param_info": self.param_info,
            "losses": losses,
            "final_metrics": {
                "val": self.best_val_res if self.val_bool else None,
                "test": self.best_test_res if self.test_bool else None,
            },
            "time_memory": self.tracker.log_final(),
            "args": {k: v for k, v in vars(self.args).items() if k not in ["paths", "ids", "logger"]},
        }

        # logging
        if self.val_bool and self.best_val_res is not None:
            self.args.logger.write('Final val res: ' + str(format_dict(self.best_val_res)))
        if self.test_bool and self.best_test_res is not None:
            self.args.logger.write('Final test res: ' + str(format_dict(self.best_test_res)))
        self.args.logger.write('Time memory metrics: ' + str(format_dict(self.final_results["time_memory"])))
        
    def train_one_epoch(self):    
        # reset counters
        self.cum_train_loss_print = 0
        self.num_batches_trained_print = 0
        self.cum_train_loss = 0
        self.num_batches_trained = 0
        
        self.model.train()
        train_bar = tqdm(range(self.num_batches_per_epoch))
    
        for _ in train_bar: 

            batch = self.batcher.get_batch()
            batch = {k: v.to(self.args.device) for k, v in batch.items()}

            # forward pass
            if self.args.train_mode == "pretrain":
                # skip empty batches
                if batch['forecast_mask'].sum() == 0:
                    self.args.logger.write('Skipped empty batch in training.')
                    continue
                loss, _ = self.model(**batch) 
            else:
                logits, _ = self.model(**batch)  
                loss = self.model.compute_loss(logits, batch['labels']) 

            # backward pass
            if not torch.isnan(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
                if (self.step_total + 1) % self.args.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()  
            
            self.tracker.update_memory_batch()     
            
            # track loss
            self.cum_train_loss_print += loss.item()
            self.cum_train_loss += loss.item()
            self.step_total += 1
            self.num_batches_trained_print += 1
            self.num_batches_trained += 1

            # print every few steps
            if self.args.print_train_loss_every is not None and self.step_total % self.args.print_train_loss_every == 0:
                self.log_train_loss(self.step_total)

            # validation
            if self.args.validate_every is not None and self.step_total % self.args.validate_every == 0:
                self.evaluate_at_time(self.step_total)
                self.model.train()        
        
        
    def evaluate_at_time(self, step):
        if self.val_bool:
            self.evaluator.evaluate(self.model, 'val', train_step=step)
        #if not self.pretrain and not self.args.grid:
        if self.test_bool:
            self.evaluator.evaluate(self.model, 'test', train_step=step)
        

    def log_train_loss(self,step):
        avg_loss = self.cum_train_loss_print / self.num_batches_trained_print
        self.args.logger.write(f'\nTrain-loss at step {step}: {avg_loss}')
        self.cum_train_loss_print = 0
        self.num_batches_trained_print = 0


    def evaluate_at_epoch(self, step):
        # Track training loss per epoch
        avg_train_loss = self.cum_train_loss / max(1, self.num_batches_trained)
        self.train_losses_per_epoch.append(avg_train_loss)
        self.cum_train_loss = 0.0
        self.num_batches_trained = 0

        # Evaluate on validation / test set if available
        val_res = self.evaluator.evaluate(self.model, 'val', train_step=step) if self.val_bool else None
        test_res = self.evaluator.evaluate(self.model, 'test', train_step=step) if self.test_bool else None

        # If training with patience
        if self.val_bool:
            # Record current loss
            self.val_losses_per_epoch.append(val_res['loss'])
            # Determine current validation metric
            if self.args.criterion == "auc":
                curr_val_metric = val_res['auprc'] + val_res['auroc']
            else:  # default is loss
                curr_val_metric = -val_res['loss']

            if curr_val_metric > self.best_val_metric:
                self._update_best_checkpoint(val_res, test_res, curr_val_metric)
            else:
                self._update_patience()
        # Last results are always the best
        else:
            self.best_model = self.model
            self.best_test_res = test_res 
            self.tracker.update_train_total()             
                            
    def _update_best_checkpoint(self, val_res, test_res, curr_val_metric):
        self.best_model = self.model
        self.best_val_metric = curr_val_metric
        self.best_val_res = val_res
        self.best_test_res = test_res
        self.args.logger.write('\nSaving ckpt at ' + self.model_path_best)
        torch.save(self.model.state_dict(), self.model_path_best)
        self.wait = self.args.patience
        self.args.logger.write('Wait still at ' + str(self.wait))
        self.tracker.update_train_total()
        
        
    def _update_patience(self):
        self.wait -= 1
        self.args.logger.write(f'Updating wait to {self.wait} based on {self.args.criterion}')
        if self.wait == 0:
            self.args.logger.write('Patience reached')
            self.patience_reached = True
