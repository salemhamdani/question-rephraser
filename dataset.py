#!/usr/bin/env python3
"""
Dataset classes for Question Rephrasing
======================================

This module contains dataset classes for handling Disfl-QA data.
"""

import json
import random
from typing import Dict, List
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class DisflQADataset(Dataset):
    """Dataset class for question rephrasing tasks (supports Disfl-QA and other formats)"""
    
    def __init__(self, data_path: str, tokenizer: PreTrainedTokenizer, config, 
                 input_prefix: str = "rephrase: ", is_train: bool = True):
        self.tokenizer = tokenizer
        self.config = config
        self.input_prefix = input_prefix
        self.is_train = is_train
        
        # Load data based on file extension
        if data_path.endswith('.json'):
            with open(data_path, 'r') as f:
                self.data = json.load(f)
        else:
            raise ValueError(f"Unsupported file format: {data_path}")
        
        # Convert to standard format
        self.examples = self._standardize_data(self.data)
        
        print(f"Loaded {len(self.examples)} examples from {data_path}")
        
        # Apply data augmentation if training
        if is_train and hasattr(config, 'use_data_augmentation') and config.use_data_augmentation:
            self.examples = self._augment_data(self.examples)
            print(f"After augmentation: {len(self.examples)} examples")
    
    def _standardize_data(self, data) -> List[Dict]:
        """Convert data to standard format"""
        examples = []
        
        if isinstance(data, dict):
            for key, value in data.items():
                examples.append({
                    'id': key,
                    'input': value.get('disfluent', value.get('input', '')),
                    'target': value.get('original', value.get('target', ''))
                })
        
        return examples
    
    def _augment_data(self, examples: List[Dict]) -> List[Dict]:
        """Apply data augmentation techniques"""
        augmented = examples.copy()
        
        for example in examples:
            if random.random() < getattr(self.config, 'augmentation_prob', 0.3):
                # Technique 1: Add noise markers
                input_text = example['input']
                noise_markers = ['um', 'uh', 'er', 'well', 'like']
                
                # Insert random noise markers
                words = input_text.split()
                noise_inserted = False
                
                for i, word in enumerate(words):
                    if random.random() < 0.2:
                        noise = random.choice(noise_markers)
                        words.insert(i, noise)
                        noise_inserted = True
                        break
                
                # Only add augmented example if noise was actually inserted
                if noise_inserted:
                    augmented.append({
                        'id': example['id'] + '_aug1',
                        'input': ' '.join(words),
                        'target': example['target']
                    })
                
                # Technique 2: Duplicate words
                words = input_text.split()
                if len(words) > 3:
                    idx = random.randint(1, len(words) - 2)
                    words.insert(idx, words[idx])
                    
                    augmented.append({
                        'id': example['id'] + '_aug2',
                        'input': ' '.join(words),
                        'target': example['target']
                    })
        
        return augmented
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Prepare input and target
        input_text = f"{self.input_prefix}{example['input']}"
        target_text = example['target']
        
        # Tokenize
        input_encoding = self.tokenizer(
            input_text,
            max_length=getattr(self.config, 'max_input_length', 512),
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        target_encoding = self.tokenizer(
            target_text,
            max_length=getattr(self.config, 'max_output_length', 128),
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            'input_ids': input_encoding.input_ids.flatten(),
            'attention_mask': input_encoding.attention_mask.flatten(),
            'labels': target_encoding.input_ids.flatten()
        } 