#!/usr/bin/env python3
"""
Data Analysis for Question Rephrasing Project
============================================

This script analyzes the Disfl-QA dataset to understand:
- Data distribution and statistics
- Question types and patterns
- Disfluency patterns and characteristics
- Text length distributions
- Data quality and preprocessing needs
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from pathlib import Path
import nltk
from nltk.tokenize import word_tokenize
    
# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

class DisflQAAnalyzer:
    def __init__(self, data_dir="data-set"):
        self.data_dir = Path(data_dir)
        self.train_data = None
        self.test_data = None
        self.stats = {}
        
    def load_data(self):
        """Load the training and test datasets"""
        print("Loading datasets...")
        
        # Load training data
        with open(self.data_dir / "train.json", 'r') as f:
            self.train_data = json.load(f)
        
        # Load test data
        with open(self.data_dir / "test.json", 'r') as f:
            self.test_data = json.load(f)
            
        print(f"Loaded {len(self.train_data)} training examples")
        print(f"Loaded {len(self.test_data)} test examples")
        
    def basic_statistics(self):
        """Calculate basic dataset statistics"""
        print("\n" + "="*50)
        print("BASIC DATASET STATISTICS")
        print("="*50)
        
        # Convert to DataFrames for easier analysis
        train_df = pd.DataFrame([
            {
                'id': k,
                'original': v['original'],
                'disfluent': v['disfluent'],
                'original_length': len(v['original']),
                'disfluent_length': len(v['disfluent']),
                'original_words': len(word_tokenize(v['original'])),
                'disfluent_words': len(word_tokenize(v['disfluent']))
            }
            for k, v in self.train_data.items()
        ])
        
        test_df = pd.DataFrame([
            {
                'id': k,
                'original': v['original'],
                'disfluent': v['disfluent'],
                'original_length': len(v['original']),
                'disfluent_length': len(v['disfluent']),
                'original_words': len(word_tokenize(v['original'])),
                'disfluent_words': len(word_tokenize(v['disfluent']))
            }
            for k, v in self.test_data.items()
        ])
        
        # Store for later use
        self.train_df = train_df
        self.test_df = test_df
        
        print(f"Training examples: {len(train_df)}")
        print(f"Test examples: {len(test_df)}")
        
        # Character length statistics
        print(f"\nCharacter Length Statistics (Training):")
        print(f"Original - Mean: {train_df['original_length'].mean():.1f}, "
              f"Std: {train_df['original_length'].std():.1f}, "
              f"Min: {train_df['original_length'].min()}, "
              f"Max: {train_df['original_length'].max()}")
        
        print(f"Disfluent - Mean: {train_df['disfluent_length'].mean():.1f}, "
              f"Std: {train_df['disfluent_length'].std():.1f}, "
              f"Min: {train_df['disfluent_length'].min()}, "
              f"Max: {train_df['disfluent_length'].max()}")
        
        # Word count statistics
        print(f"\nWord Count Statistics (Training):")
        print(f"Original - Mean: {train_df['original_words'].mean():.1f}, "
              f"Std: {train_df['original_words'].std():.1f}")
        print(f"Disfluent - Mean: {train_df['disfluent_words'].mean():.1f}, "
              f"Std: {train_df['disfluent_words'].std():.1f}")
        
        return train_df, test_df
    
    def analyze_disfluency_patterns(self):
        """Analyze common disfluency patterns in the data"""
        print("\n" + "="*50)
        print("DISFLUENCY PATTERN ANALYSIS")
        print("="*50)
        
        # Dynamic disfluency marker detection
        all_words = Counter()
        disfluent_only_words = Counter()
        
        for item in self.train_data.values():
            original_words = set(word_tokenize(item['original'].lower()))
            disfluent_words = word_tokenize(item['disfluent'].lower())
            
            # Count all words in disfluent text
            all_words.update(disfluent_words)
            
            # Find words that appear only in disfluent text
            disfluent_only = set(disfluent_words) - original_words
            disfluent_only_words.update(disfluent_only)
        
        # Filter out common words and short words to focus on disfluency markers
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'can', 'may', 'might', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their'}
        
        # Get potential disfluency markers (words that appear frequently in disfluent text but not in original)
        potential_markers = {word: count for word, count in disfluent_only_words.most_common(50) 
                           if word not in common_words and len(word) > 1}
        
        # Also look for common patterns in the data
        common_patterns = []
        for item in self.train_data.values():
            disfluent = item['disfluent'].lower()
            # Look for common disfluency patterns
            patterns = ['um', 'uh', 'er', 'hmm', 'oh', 'well', 'you know', 'i mean', 'like', 'actually', 'basically', 'literally', 'sort of', 'kind of', 'right', 'okay', 'so', 'now', 'then', 'wait', 'sorry', 'oops', 'nevermind', 'scratch that', 'make that', 'better', 'correction', 'rather', 'instead']
            for pattern in patterns:
                if pattern in disfluent:
                    common_patterns.append(pattern)
        
        pattern_counts = Counter(common_patterns)
        
        print("Top 15 Dynamic Disfluency Markers:")
        for marker, count in list(potential_markers.items())[:15]:
            percentage = (count / len(self.train_data)) * 100
            print(f"'{marker}': {count} ({percentage:.1f}%)")
        
        print(f"\nTop 10 Common Disfluency Patterns:")
        for pattern, count in pattern_counts.most_common(10):
            percentage = (count / len(self.train_data)) * 100
            print(f"'{pattern}': {count} ({percentage:.1f}%)")
        
        # Analyze question patterns
        print(f"\nQuestion Word Analysis:")
        question_words = ['what', 'when', 'where', 'who', 'why', 'how', 'which']
        question_patterns = {}
        
        for qword in question_words:
            count = sum(1 for item in self.train_data.values() 
                       if item['original'].lower().startswith(qword))
            percentage = (count / len(self.train_data)) * 100
            question_patterns[qword] = {'count': count, 'percentage': round(percentage, 1)}
            print(f"Questions starting with '{qword}': {count} ({percentage:.1f}%)")
        
        return {
            'dynamic_markers': dict(list(potential_markers.items())[:15]),
            'common_patterns': dict(pattern_counts.most_common(10)),
            'question_patterns': question_patterns
        }
    
    def analyze_edit_operations(self):
        """Analyze the types of edit operations needed"""
        print("\n" + "="*50)
        print("EDIT OPERATIONS ANALYSIS")
        print("="*50)
        
        operations = {
            'deletions': 0,
            'insertions': 0,
            'substitutions': 0,
            'reorderings': 0
        }
        
        for item in self.train_data.values():
            original_words = word_tokenize(item['original'].lower())
            disfluent_words = word_tokenize(item['disfluent'].lower())
            
            original_set = set(original_words)
            disfluent_set = set(disfluent_words)
            
            # Words only in disfluent (need deletion)
            only_disfluent = disfluent_set - original_set
            if only_disfluent:
                operations['deletions'] += 1
            
            # Words only in original (need insertion)  
            only_original = original_set - disfluent_set
            if only_original:
                operations['insertions'] += 1
            
            # Detect substitutions (words that changed)
            # Find words that appear in both but might be different
            common_words = original_set & disfluent_set
            if len(original_words) == len(disfluent_words):
                # Same length - check if words changed
                if original_words != disfluent_words:
                    operations['substitutions'] += 1
            elif len(common_words) <= min(len(original_words), len(disfluent_words)):
                # Different lengths but some words changed
                operations['substitutions'] += 1
            
            # Detect reorderings (same words, different order)
            if (len(original_words) == len(disfluent_words) and 
                original_set == disfluent_set and 
                original_words != disfluent_words):
                operations['reorderings'] += 1
        
        print("Edit Operation Frequency:")
        edit_operations = {}
        for op, count in operations.items():
            edit_operations[op] = {'count': count}
            print(f"{op.capitalize()}: {count}")
        
        return edit_operations
    
    def visualize_distributions(self):
        """Create visualizations of data distributions"""
        print("\n" + "="*50)
        print("CREATING VISUALIZATIONS")
        print("="*50)
        
        output_dir = Path("data-analysis")
        output_dir.mkdir(exist_ok=True)
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Question Rephrasing Dataset Analysis', fontsize=16, fontweight='bold')
        
        # 1. Length distribution comparison
        axes[0, 0].hist(self.train_df['original_length'], bins=50, alpha=0.7, 
                       label='Original', density=True)
        axes[0, 0].hist(self.train_df['disfluent_length'], bins=50, alpha=0.7, 
                       label='Disfluent', density=True)
        axes[0, 0].set_xlabel('Character Length')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].set_title('Character Length Distribution')
        axes[0, 0].legend()
        
        # 2. Word count distribution
        axes[0, 1].hist(self.train_df['original_words'], bins=50, alpha=0.7, 
                       label='Original', density=True)
        axes[0, 1].hist(self.train_df['disfluent_words'], bins=50, alpha=0.7, 
                       label='Disfluent', density=True)
        axes[0, 1].set_xlabel('Word Count')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].set_title('Word Count Distribution')
        axes[0, 1].legend()
        
        # 3. Average word length distribution
        avg_word_length_original = self.train_df['original_length'] / self.train_df['original_words'].replace(0, 1)
        avg_word_length_disfluent = self.train_df['disfluent_length'] / self.train_df['disfluent_words'].replace(0, 1)
        axes[0, 2].hist(avg_word_length_original, bins=50, alpha=0.7, color='blue', label='Original', density=True)
        axes[0, 2].hist(avg_word_length_disfluent, bins=50, alpha=0.7, color='orange', label='Disfluent', density=True)
        axes[0, 2].set_xlabel('Average Word Length (chars)')
        axes[0, 2].set_ylabel('Density')
        axes[0, 2].set_title('Average Word Length Distribution')
        axes[0, 2].legend()
        
        # 4. Scatter plot: original vs disfluent length
        axes[1, 0].scatter(self.train_df['original_length'], 
                          self.train_df['disfluent_length'], alpha=0.3)
        axes[1, 0].plot([0, 300], [0, 300], 'r--', label='Equal length')
        axes[1, 0].set_xlabel('Original Length (chars)')
        axes[1, 0].set_ylabel('Disfluent Length (chars)')
        axes[1, 0].set_title('Length Correlation')
        axes[1, 0].legend()
        
        # 5. Word count scatter
        axes[1, 1].scatter(self.train_df['original_words'], 
                          self.train_df['disfluent_words'], alpha=0.3)
        axes[1, 1].plot([0, 50], [0, 50], 'r--', label='Equal count')
        axes[1, 1].set_xlabel('Original Words')
        axes[1, 1].set_ylabel('Disfluent Words')
        axes[1, 1].set_title('Word Count Correlation')
        axes[1, 1].legend()
        
        # 6. Length difference distribution
        length_diff = self.train_df['disfluent_length'] - self.train_df['original_length']
        axes[1, 2].hist(length_diff, bins=50, alpha=0.7, color='purple')
        axes[1, 2].set_xlabel('Length Difference (chars)')
        axes[1, 2].set_ylabel('Frequency')
        axes[1, 2].set_title('Length Difference Distribution')
        axes[1, 2].axvline(x=0, color='red', linestyle='--', label='No difference')
        axes[1, 2].legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / 'dataset_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Visualizations saved to {output_dir}/dataset_analysis.png")
    
    def save_analysis_report(self, disfluency_patterns=None, edit_operations=None):
        """Save a comprehensive analysis report"""
        print("\n" + "="*50)
        print("SAVING ANALYSIS REPORT")
        print("="*50)
        
        output_dir = Path("data-analysis")
        output_dir.mkdir(exist_ok=True)
        
        report = {
            'dataset_overview': {
                'train_size': len(self.train_data),
                'test_size': len(self.test_data),
                'total_size': len(self.train_data) + len(self.test_data)
            },
            'length_statistics': {
                'train': {
                    'original_chars': {
                        'mean': float(self.train_df['original_length'].mean()),
                        'std': float(self.train_df['original_length'].std()),
                        'min': int(self.train_df['original_length'].min()),
                        'max': int(self.train_df['original_length'].max())
                    },
                    'disfluent_chars': {
                        'mean': float(self.train_df['disfluent_length'].mean()),
                        'std': float(self.train_df['disfluent_length'].std()),
                        'min': int(self.train_df['disfluent_length'].min()),
                        'max': int(self.train_df['disfluent_length'].max())
                    }
                }
            }
        }
        
        if disfluency_patterns:
            report['disfluency_patterns'] = {
                'dynamic_markers': {
                    '_comment': 'Words that appear ONLY in disfluent text but NOT in original text (automatically discovered)',
                    'data': disfluency_patterns.get('dynamic_markers', {})
                },
                'common_patterns': {
                    '_comment': 'All occurrences of known disfluency patterns found in disfluent text (predefined patterns)',
                    'data': disfluency_patterns.get('common_patterns', {})
                },
                'question_patterns': disfluency_patterns.get('question_patterns', {})
            }
        
        if edit_operations:
            report['edit_operations'] = edit_operations
        
        report_path = output_dir / 'analysis_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Analysis report saved to {report_path}")
    
    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        print("Starting comprehensive data analysis...")
        
        self.load_data()
        self.basic_statistics()
        disfluency_patterns = self.analyze_disfluency_patterns()
        edit_operations = self.analyze_edit_operations()
        self.visualize_distributions()
        self.save_analysis_report(disfluency_patterns, edit_operations)
        
        print("\n" + "="*50)
        print("ANALYSIS COMPLETE!")
        print("="*50)

if __name__ == "__main__":
    analyzer = DisflQAAnalyzer()
    analyzer.run_complete_analysis() 