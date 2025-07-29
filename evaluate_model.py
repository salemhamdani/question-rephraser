#!/usr/bin/env python3
"""
Model Evaluation on Test Set
============================

Simple script to evaluate trained model on test.json and log metrics.
"""

import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List
import random
from transformers import (
    T5ForConditionalGeneration, T5Tokenizer,
    BartForConditionalGeneration, BartTokenizer,
    EncoderDecoderModel, BertTokenizer
)
import logging
from tqdm import tqdm
import argparse
import evaluate

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Simple model evaluation on test set"""
    
    def __init__(self, model_path: str, data_path: str = "data-set"):
        self.model_path = Path(model_path)
        self.data_path = Path(data_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model and tokenizer
        self.model, self.tokenizer = self._load_model_and_tokenizer()
        
        # Load test data
        self._load_test_data()
        
        # Load metrics
        self._setup_metrics()
        
    def _load_model_and_tokenizer(self):
        """Load the trained model and tokenizer"""
        try:
            # Extract model name directly from directory path
            dir_name = str(self.model_path.name).lower()
            
            # Simple model type detection from directory name
            if 't5' in dir_name:
                model_type = 't5'
                tokenizer = T5Tokenizer.from_pretrained(self.model_path)
                model = T5ForConditionalGeneration.from_pretrained(self.model_path)
            elif 'bart' in dir_name:
                model_type = 'bart'
                tokenizer = BartTokenizer.from_pretrained(self.model_path)
                model = BartForConditionalGeneration.from_pretrained(self.model_path)
            elif 'bert' in dir_name:
                model_type = 'bert'
                tokenizer = BertTokenizer.from_pretrained(self.model_path)
                model = EncoderDecoderModel.from_pretrained(self.model_path)
            else:
                raise ValueError(f"Unknown model type: {dir_name}")
                        
            model.to(self.device)
            model.eval()
            
            logger.info(f"Successfully loaded {model_type} model from {self.model_path}")
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            logger.error(f"Model path: {self.model_path}")
            logger.error(f"Available files: {list(self.model_path.glob('*'))}")
            raise
    
    def _load_test_data(self):
        """Load test dataset"""
        test_path = self.data_path / "test.json"
        
        with open(test_path) as f:
            self.test_data = json.load(f)
        
        # Convert to list format
        self.test_examples = [
            {'id': k, 'disfluent': v['disfluent'], 'original': v['original']}
            for k, v in self.test_data.items()
        ]
        
        logger.info(f"Loaded {len(self.test_examples)} test examples")
    
    def _setup_metrics(self):
        """Setup evaluation metrics"""
        self.metrics = {
            'bleu': evaluate.load('bleu'),
            'rouge': evaluate.load('rouge')
        }
    
    def evaluate_model(self, max_samples: int = None):
        """Evaluate model on test set"""
        logger.info("Evaluating model on test set...")
        
        # Sample if requested
        examples = self.test_examples
        if max_samples and len(examples) > max_samples:
            examples = random.sample(examples, max_samples)
        
        predictions = []
        references = []
        losses = []
        
        with torch.no_grad():
            for example in tqdm(examples, desc="Evaluating"):
                try:
                    # Prepare input
                    input_text = f"rephrase: {example['disfluent']}"
                    target_text = example['original']
                    
                    # Tokenize input
                    inputs = self.tokenizer(
                        input_text,
                        max_length=512,
                        padding=True,
                        truncation=True,
                        return_tensors="pt"
                    ).to(self.device)
                    
                    # Generate prediction
                    outputs = self.model.generate(
                        **inputs,
                        max_length=128,
                        num_beams=4,
                        early_stopping=True
                    )
                    
                    prediction = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    
                    # Compute perplexity
                    targets = self.tokenizer(
                        target_text,
                        max_length=128,
                        padding=True,
                        truncation=True,
                        return_tensors="pt"
                    ).to(self.device)
                    
                    loss_outputs = self.model(**inputs, labels=targets.input_ids)
                    loss = loss_outputs.loss.item()
                    
                    predictions.append(prediction)
                    references.append(target_text)
                    losses.append(loss)
                    
                except Exception as e:
                    logger.warning(f"Error evaluating example: {e}")
                    continue
        
        # Compute metrics
        results = self._compute_metrics(predictions, references, losses)
        
        return results
    
    def _compute_metrics(self, predictions: List[str], references: List[str], losses: List[float]) -> Dict:
        """Compute evaluation metrics"""
        results = {}
        
        # Perplexity
        if losses:
            results['perplexity'] = torch.exp(torch.mean(torch.tensor(losses))).item()
        else:
            results['perplexity'] = 0.0
        
        bleu_result = self.metrics['bleu'].compute(
            predictions=predictions, 
            references=[[ref] for ref in references]
        )
        results['bleu'] = bleu_result['bleu']
        
        rouge_result = self.metrics['rouge'].compute(
            predictions=predictions, 
            references=references
        )
        results['rouge1'] = rouge_result['rouge1']
        results['rouge2'] = rouge_result['rouge2']
        results['rougeL'] = rouge_result['rougeL']
        
        # Exact match accuracy
        exact_matches = sum(1 for p, r in zip(predictions, references) 
                           if p.strip().lower() == r.strip().lower())
        results['exact_match'] = exact_matches / len(predictions)
        
        # General accuracy
        results['accuracy'] = results['exact_match']
        
        # Character-level accuracy
        if predictions:
            results['char_accuracy'] = sum(sum(1 for pc, rc in zip(p, r) if pc == rc) for p, r in zip(predictions, references)) / sum(max(len(p), len(r)) for p, r in zip(predictions, references))
        else:
            results['char_accuracy'] = 0.0
        
        # Word-level accuracy
        if predictions:
            results['word_accuracy'] = sum(sum(1 for pw, rw in zip(p.split(), r.split()) if pw == rw) for p, r in zip(predictions, references)) / sum(max(len(p.split()), len(r.split())) for p, r in zip(predictions, references))
        else:
            results['word_accuracy'] = 0.0
        
        return results
    
    def generate_report(self, max_samples: int = None):
        """Generate evaluation report"""
        logger.info("Generating evaluation report...")
        
        # Evaluate model
        results = self.evaluate_model(max_samples)
        
        # Create report
        report = {
            'model_path': str(self.model_path),
            'test_samples': len(self.test_examples),
            'evaluated_samples': max_samples or len(self.test_examples),
            'metrics': results
        }
        
        # Save report
        report_path = self.model_path / 'test_evaluation.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        print("\n" + "="*50)
        print("TEST SET EVALUATION RESULTS")
        print("="*50)
        print(f"Samples evaluated: {report['evaluated_samples']}")
        print(f"Perplexity: {results['perplexity']:.2f}")
        print(f"BLEU: {results['bleu']:.4f}")
        print(f"ROUGE-1: {results['rouge1']:.4f}")
        print(f"ROUGE-2: {results['rouge2']:.4f}")
        print(f"ROUGE-L: {results['rougeL']:.4f}")
        print(f"Accuracy: {results['accuracy']:.1%}")
        print(f"Character Accuracy: {results['char_accuracy']:.1%}")
        print(f"Word Accuracy: {results['word_accuracy']:.1%}")
        print(f"Report saved to: {report_path}")
        print("="*50)
        
        return report

def main():
    parser = argparse.ArgumentParser(description="Evaluate model on test set")
    parser.add_argument("model_path", help="Path to trained model directory")
    parser.add_argument("--data_path", default="data-set", help="Path to dataset directory")
    parser.add_argument("--max_samples", type=int, help="Maximum samples to evaluate")
    
    args = parser.parse_args()
    
    # Run evaluation
    evaluator = ModelEvaluator(args.model_path, args.data_path)
    evaluator.generate_report(args.max_samples)

if __name__ == "__main__":
    main() 