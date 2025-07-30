#!/usr/bin/env python3
"""
Question Rephrasing Model Trainer
=================================

This script implements multiple approaches for question rephrasing:
1. Transformer seq2seq models (T5, BART)
2. BERT-based encoder-decoder
3. Custom transformer architectures
4. Data augmentation techniques
"""

import torch
from transformers import (
    T5ForConditionalGeneration, T5Tokenizer,
    BartForConditionalGeneration, BartTokenizer,
    BertTokenizer, EncoderDecoderModel)
import numpy as np
import random
import argparse
import logging
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from dataclasses import dataclass, field
from typing import List, Dict
import evaluate
import json
from pathlib import Path

# Import dataset classes
from dataset import DisflQADataset

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for model training"""
    model_name: str = "t5-small" # t5-small, t5-base, t5-large, t5-3b, t5-11b, facebook/bart-base, facebook/bart-large, bert-base-uncased, bert-large-uncased, bert-base-cased
    max_input_length: int = 512
    max_output_length: int = 128
    batch_size: int = 16
    num_epochs: int = 3
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    gradient_accumulation_steps: int = 1
    save_strategy: str = "epoch"
    evaluation_strategy: str = "epoch"
    load_best_model_at_end: bool = True
    seed: int = 42
    # Data augmentation settings
    use_data_augmentation: bool = True
    augmentation_prob: float = 0.3
    # Configurable metrics for LLMs
    metrics: List[str] = field(default_factory=lambda: ['bleu', 'gleu', 'rouge', 'accuracy'])  # List of metrics to compute (bleu, gleu, rouge, accuracy, perplexity, etc.)
    # Configurable loss function for LLMs
    loss_function: str = "cross_entropy"  # cross_entropy, label_smoothing
    label_smoothing: float = 0.0  # For label smoothing loss
    # Early stopping configuration
    use_early_stopping: bool = False  # Whether to use early stopping
    early_stopping_patience: int = 3  # Number of epochs to wait before stopping

class QuestionRephraserLightningModule(L.LightningModule):
    """Lightning module for question rephrasing"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.save_hyperparameters()
        self.model, self.tokenizer = self._load_model_and_tokenizer()
        self._load_datasets()
        self._setup_metrics()
        self._setup_loss_function()
    
    def _load_model_and_tokenizer(self):
        """Load the model and tokenizer"""
        if 't5' in self.config.model_name.lower():
            tokenizer = T5Tokenizer.from_pretrained(self.config.model_name)
            model = T5ForConditionalGeneration.from_pretrained(self.config.model_name)
        elif 'bart' in self.config.model_name.lower():
            tokenizer = BartTokenizer.from_pretrained(self.config.model_name)
            model = BartForConditionalGeneration.from_pretrained(self.config.model_name)
        elif 'bert' in self.config.model_name.lower():
            tokenizer = BertTokenizer.from_pretrained(self.config.model_name)
            model = EncoderDecoderModel.from_encoder_decoder_pretrained(
                self.config.model_name, self.config.model_name
            )
            # Set special tokens for generation
            model.config.decoder_start_token_id = tokenizer.cls_token_id
            model.config.bos_token_id = tokenizer.cls_token_id
            model.config.eos_token_id = tokenizer.sep_token_id
            model.config.pad_token_id = tokenizer.pad_token_id
            
            # Also set on generation_config if it exists
            if hasattr(model, 'generation_config'):
                model.generation_config.decoder_start_token_id = tokenizer.cls_token_id
                model.generation_config.bos_token_id = tokenizer.cls_token_id
                model.generation_config.eos_token_id = tokenizer.sep_token_id
                model.generation_config.pad_token_id = tokenizer.pad_token_id
            
            # Set on decoder config for encoder-decoder models
            if hasattr(model.config, 'decoder') and model.config.decoder:
                model.config.decoder.decoder_start_token_id = tokenizer.cls_token_id
                model.config.decoder.bos_token_id = tokenizer.cls_token_id
                model.config.decoder.eos_token_id = tokenizer.sep_token_id
                model.config.decoder.pad_token_id = tokenizer.pad_token_id
            
            # Debug: print the token IDs to verify they're set
            print(f"🔧 BERT token setup: decoder_start={model.config.decoder_start_token_id}, "
                  f"bos={model.config.bos_token_id}, eos={model.config.eos_token_id}, "
                  f"pad={model.config.pad_token_id}")
            print(f"🔧 CLS token ID: {tokenizer.cls_token_id}, SEP token ID: {tokenizer.sep_token_id}")
            
        else:
            raise ValueError(f"Unsupported model: {self.config.model_name}")
        
        return model, tokenizer
    
    def _load_datasets(self):
        """Load train and validation datasets"""
        # Create a simple config for datasets
        dataset_config = type('Config', (), {
            'use_data_augmentation': True,
            'augmentation_prob': 0.3
        })()
        
        # Load data
        train_data = DisflQADataset("data-set/train.json", self.tokenizer, dataset_config, is_train=True)
        eval_data = DisflQADataset("data-set/dev.json", self.tokenizer, dataset_config, is_train=False)
        
        self.train_dataset = train_data
        self.eval_dataset = eval_data
    
    def forward(self, batch):
        """Forward pass"""
        return self.model(**batch)
    
    def training_step(self, batch, batch_idx):
        """Training step"""
        outputs = self(batch)
        
        # Use Cross-Entropy Loss
        loss = self.loss_fn(outputs.logits.view(-1, outputs.logits.size(-1)), batch['labels'].view(-1))
        
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step"""
        outputs = self(batch)
        
        # Use Cross-Entropy Loss
        loss = self.loss_fn(outputs.logits.view(-1, outputs.logits.size(-1)), batch['labels'].view(-1))
        
        self.log('val_loss', loss, prog_bar=True)
        
        # Generate predictions for metrics
        generated_ids = self.model.generate(
            batch['input_ids'],
            attention_mask=batch['attention_mask'],
            max_length=self.config.max_output_length,
            num_beams=4,
            early_stopping=True
        )
        
        # Decode predictions and references
        predictions = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        
        # Fix reference decoding - replace -100 with pad token before decoding
        labels_for_decode = batch['labels'].clone()
        labels_for_decode[labels_for_decode == -100] = self.tokenizer.pad_token_id
        references = self.tokenizer.batch_decode(labels_for_decode, skip_special_tokens=True)
        
        # Store for epoch end
        self.validation_step_outputs.append({
            'predictions': predictions,
            'references': references,
            'loss': loss.item()
        })
        
        return loss
    
    def on_validation_epoch_start(self):
        """Initialize validation outputs"""
        self.validation_step_outputs = []
    
    def on_validation_epoch_end(self):
        """Compute metrics at epoch end"""
        if not self.validation_step_outputs:
            return
        
        # Collect all predictions and references
        all_predictions = []
        all_references = []
        all_losses = []
        
        for output in self.validation_step_outputs:
            all_predictions.extend(output['predictions'])
            all_references.extend(output['references'])
            all_losses.append(output['loss'])
        
        # Compute metrics
        metrics = self._compute_metrics(all_predictions, all_references, all_losses)
        
        # Log metrics
        for metric_name, metric_value in metrics.items():
            self.log(f'val_{metric_name}', metric_value, prog_bar=True)
        
        # Calculate and log perplexity
        if all_losses:
            # Convert float losses to tensor for calculation
            perplexity = torch.exp(torch.mean(torch.tensor(all_losses))).item()
            self.log('val_perplexity', perplexity, prog_bar=True)
    
    def configure_optimizers(self):
        """Configure optimizer"""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        return optimizer
    
    def train_dataloader(self):
        """Training data loader"""
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
    
    def val_dataloader(self):
        """Validation data loader"""
        return torch.utils.data.DataLoader(
            self.eval_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

    def _setup_metrics(self):
        """Setup evaluation metrics for LLMs"""
        self.metrics = {}
        
        # Default metrics if none specified
        if self.config.metrics is None:
            self.config.metrics = ['bleu', 'gleu', 'rouge', 'accuracy', 'perplexity']
        
        # Load requested metrics
        for metric_name in self.config.metrics:
            if metric_name == 'bleu':
                self.metrics['bleu'] = evaluate.load('bleu')
            elif metric_name == 'gleu':
                self.metrics['gleu'] = evaluate.load('google_bleu')
            elif metric_name == 'rouge':
                self.metrics['rouge'] = evaluate.load('rouge')
            elif metric_name == 'sacrebleu':
                self.metrics['sacrebleu'] = evaluate.load('sacrebleu')
            elif metric_name == 'meteor':
                self.metrics['meteor'] = evaluate.load('meteor')
            elif metric_name == 'accuracy':
                # Custom accuracy metric for text generation
                self.metrics['accuracy'] = lambda preds, refs: sum(1 for p, r in zip(preds, refs) if p.strip().lower() == r.strip().lower()) / len(preds) if preds else 0
            elif metric_name == 'perplexity':
                # Custom perplexity metric
                self.metrics['perplexity'] = lambda preds, refs, losses: torch.exp(torch.mean(torch.stack(losses))).item() if losses else 0.0
            elif metric_name == 'exact_match':
                # Custom exact match metric
                self.metrics['exact_match'] = lambda preds, refs: sum(1 for p, r in zip(preds, refs) if p.strip().lower() == r.strip().lower()) / len(preds) if preds else 0
            elif metric_name == 'char_accuracy':
                # Character-level accuracy
                self.metrics['char_accuracy'] = lambda preds, refs: sum(sum(1 for pc, rc in zip(p, r) if pc == rc) for p, r in zip(preds, refs)) / sum(max(len(p), len(r)) for p, r in zip(preds, refs)) if preds else 0
            elif metric_name == 'word_accuracy':
                # Word-level accuracy
                self.metrics['word_accuracy'] = lambda preds, refs: sum(sum(1 for pw, rw in zip(p.split(), r.split()) if pw == rw) for p, r in zip(preds, refs)) / sum(max(len(p.split()), len(r.split())) for p, r in zip(preds, refs)) if preds else 0
    
    def _setup_loss_function(self):
        """Setup loss function for LLMs"""
        if self.config.loss_function == "label_smoothing":
            self.loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=self.config.label_smoothing, ignore_index=-100)
        else:  # Default Cross-Entropy Loss for LLMs
            self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
    
    def _compute_metrics(self, predictions: List[str], references: List[str], losses: List[float] = None) -> Dict[str, float]:
        """Compute evaluation metrics for LLMs"""
        metrics_results = {}
        
        for metric_name, metric_fn in self.metrics.items():
            try:
                if metric_name == 'bleu':
                    result = metric_fn.compute(predictions=predictions, references=[[ref] for ref in references])
                    metrics_results['bleu'] = result['bleu']
                elif metric_name == 'gleu':
                    result = metric_fn.compute(predictions=predictions, references=[[ref] for ref in references])
                    metrics_results['gleu'] = result['google_bleu']
                elif metric_name == 'rouge':
                    result = metric_fn.compute(predictions=predictions, references=references)
                    metrics_results['rouge1'] = result['rouge1']
                    metrics_results['rouge2'] = result['rouge2']
                    metrics_results['rougeL'] = result['rougeL']
                elif metric_name == 'sacrebleu':
                    result = metric_fn.compute(predictions=predictions, references=[[ref] for ref in references])
                    metrics_results['sacrebleu'] = result['score']
                elif metric_name == 'meteor':
                    result = metric_fn.compute(predictions=predictions, references=[[ref] for ref in references])
                    metrics_results['meteor'] = result['meteor']
                elif metric_name == 'accuracy':
                    metrics_results['accuracy'] = metric_fn(predictions, references)
                elif metric_name == 'perplexity':
                    if losses:
                        metrics_results['perplexity'] = metric_fn(predictions, references, losses)
                    else:
                        metrics_results['perplexity'] = 0.0
                elif metric_name == 'exact_match':
                    metrics_results['exact_match'] = metric_fn(predictions, references)
                elif metric_name == 'char_accuracy':
                    metrics_results['char_accuracy'] = metric_fn(predictions, references)
                elif metric_name == 'word_accuracy':
                    metrics_results['word_accuracy'] = metric_fn(predictions, references)
            except Exception as e:
                logger.warning(f"Error computing {metric_name}: {e}")
                metrics_results[metric_name] = 0.0
        
        return metrics_results

class QuestionRephraserTrainer:
    """Main trainer class for question rephrasing models"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Set random seeds
        self._set_seeds()
        
        # Load datasets
        self.train_dataset = DisflQADataset(
            "data-set/train.json", None, config, is_train=True
        )
        self.eval_dataset = DisflQADataset(
            "data-set/dev.json", None, config, is_train=False
        )
        
    def _set_seeds(self):
        """Set random seeds for reproducibility"""
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)
    
    def train(self, output_dir: str, use_wandb: bool = False):
        """Train the model using PyTorch Lightning"""
        logger.info("Starting training...")
        
        # Create Lightning module
        model = QuestionRephraserLightningModule(self.config)
        
        # Setup TensorBoard logging
        tb_logger = TensorBoardLogger(
            save_dir=output_dir,
            name="lightning_logs",
            version=None
        )
        
        # Setup callbacks
        checkpoint_callback = ModelCheckpoint(
            dirpath=output_dir,
            filename='model-{epoch:02d}-{val_loss:.2f}',
            monitor='val_loss',
            mode='min',
            save_top_k=1,  # Don't save top k models
            save_last=False  # Don't save the last checkpoint
        )
        
        callbacks = [checkpoint_callback]
        
        # Add early stopping if enabled
        if self.config.use_early_stopping:
            early_stopping = EarlyStopping(
                monitor="val_loss",
                patience=self.config.early_stopping_patience,
                mode="min"
            )
            callbacks.append(early_stopping)
            logger.info(f"Early stopping enabled: patience={self.config.early_stopping_patience}")
        else:
            logger.info("Early stopping disabled")
        
        # Setup trainer
        trainer = L.Trainer(
            max_epochs=self.config.num_epochs,
            accelerator='auto',
            devices=1,
            logger=tb_logger,
            callbacks=callbacks,
            log_every_n_steps=50,
            val_check_interval=0.25
        )
        
        # Train the model
        trainer.fit(model)
        
        model.model.save_pretrained(output_dir)
        model.tokenizer.save_pretrained(output_dir)
        
        logger.info(f"Training completed. Model saved to {output_dir}")
        return output_dir

def main():
    parser = argparse.ArgumentParser(description="Train question rephrasing model")
    parser.add_argument("--model", default="t5-small", help="Model name")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--output_dir", default="models/question-rephraser", help="Output directory")
    parser.add_argument("--wandb", action="store_true", help="Use Weights & Biases logging")
    parser.add_argument("--max_input_length", type=int, default=512, help="Maximum input sequence length")
    parser.add_argument("--max_output_length", type=int, default=128, help="Maximum output sequence length")
    parser.add_argument("--use_data_augmentation", action="store_true", default=True, help="Use data augmentation")
    parser.add_argument("--augmentation_prob", type=float, default=0.3, help="Data augmentation probability")
    parser.add_argument("--metrics", nargs="+", default=["bleu", "gleu", "rouge", "accuracy", "perplexity"], 
                       help="Metrics to compute (bleu, gleu, rouge, sacrebleu, meteor, accuracy, perplexity, exact_match, char_accuracy, word_accuracy)")
    parser.add_argument("--loss_function", default="cross_entropy", 
                       choices=["cross_entropy", "label_smoothing"],
                       help="Loss function to use")
    parser.add_argument("--label_smoothing", type=float, default=0.0, help="Label smoothing factor")
    parser.add_argument("--use_early_stopping", action="store_true", help="Use early stopping")
    parser.add_argument("--early_stopping_patience", type=int, default=3, help="Early stopping patience")
    
    args = parser.parse_args()
    
    # Create configuration
    config = ModelConfig(
        model_name=args.model,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        max_input_length=args.max_input_length,
        max_output_length=args.max_output_length,
        use_data_augmentation=args.use_data_augmentation,
        augmentation_prob=args.augmentation_prob,
        metrics=args.metrics,
        loss_function=args.loss_function,
        label_smoothing=args.label_smoothing,
        use_early_stopping=args.use_early_stopping,
        early_stopping_patience=args.early_stopping_patience
    )
    
    # Create trainer
    trainer = QuestionRephraserTrainer(config)
    
    # Train model
    trainer.train(args.output_dir, args.wandb)

if __name__ == "__main__":
    main() 