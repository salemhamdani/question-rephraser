#!/usr/bin/env python3
"""
Question Rephrasing Experiment Runner

"""

import subprocess
import sys
import json
import time
from pathlib import Path
import argparse
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ExperimentRunner:
    """Main experiment runner for question rephrasing project"""
    
    def __init__(self, config_path: str = None, output_dir: str = None):
        self.start_time = datetime.now()
        
        # Load experiment configuration
        if config_path and Path(config_path).exists():
            with open(config_path) as f:
                self.config = json.load(f)
        else:
            self.config = self._get_default_config()
        
        # Create experiment directory
        if output_dir:
            self.experiments_dir = Path(output_dir)
        else:
            # Default to experiments
            self.experiments_dir = Path("experiments")
        
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
            
        logger.info(f"Experiment started at {self.start_time}")
        logger.info(f"Results will be saved to {self.experiments_dir}")
    
    def _get_default_config(self):
        """Get default experiment configuration"""
        return {
            "models_to_test": ["t5-small"],
            "batch_sizes": [8],
            "learning_rates": [5e-5],
            "epochs": [3],
            "run_test_evaluation": True,
            "use_early_stopping": True,
            "early_stopping_patience": 3
        }
    
    def train_model(self, model_name: str, batch_size: int, lr: float, epochs: int):
        """Train a single model configuration"""
        experiment_name = f"{model_name}_bs{batch_size}_lr{lr}_ep{epochs}"
        logger.info(f"Training model: {experiment_name}")
        
        # Save model directly in experiment directory
        model_output_dir = self.experiments_dir / experiment_name
        model_output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Prepare training command
            cmd = [
                sys.executable, "model_trainer.py",
                "--model", model_name,
                "--batch_size", str(batch_size),
                "--lr", str(lr),
                "--epochs", str(epochs),
                "--output_dir", str(model_output_dir)
            ]
            
            # Add early stopping configuration from config
            if self.config.get('use_early_stopping', False):
                cmd.extend(["--use_early_stopping"])
                patience = self.config.get('early_stopping_patience', 3)
                cmd.extend(["--early_stopping_patience", str(patience)])
            
            # Run training with improved error capture
            start_time = time.time()
            logger.info(f"Executing command: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                check=True,
            )
            training_time = time.time() - start_time
            

            
            logger.info(f"✅ Training completed for {experiment_name} in {training_time:.2f}s")
            return str(model_output_dir)
            
        except subprocess.TimeoutExpired as e:
            logger.error(f"⏰ Training timed out for {experiment_name}")

            return None
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Training failed for {experiment_name}: {e}")
            logger.error(f"Training stdout: {e.stdout}")
            logger.error(f"Training stderr: {e.stderr}")
            return None
            
        except Exception as e:
            logger.error(f"💥 Unexpected error during training {experiment_name}: {e}")
            return None
    
    def run_test_evaluation(self, model_path: str, experiment_name: str):
        """Run test evaluation for a trained model"""
        logger.info(f"Running test evaluation for {experiment_name}")
        
        try:
            cmd = [
                sys.executable, "evaluate_model.py",
                model_path
            ]
            
            subprocess.run(cmd, capture_output=True, text=True, check=True)      
            logger.info(f"✅ Test evaluation completed for {experiment_name}")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Test evaluation failed for {experiment_name}: {e}")

    
    def run_full_experiment(self):
        """Run the complete experimental pipeline"""
        logger.info("Starting full experimental pipeline...")
        
        for model_name in self.config['models_to_test']:
            for batch_size in self.config['batch_sizes']:
                for lr in self.config['learning_rates']:
                    for epochs in self.config['epochs']:
                        experiment_name = f"{model_name}_bs{batch_size}_lr{lr}_ep{epochs}"
                        
                        # Train model
                        model_path = self.train_model(model_name, batch_size, lr, epochs)
                        
                        if model_path:
                            # Run test evaluation
                            if self.config.get('run_test_evaluation', True):
                                self.run_test_evaluation(model_path, experiment_name)

def main():
    parser = argparse.ArgumentParser(description="Run question rephrasing experiments")
    parser.add_argument("--config", help="Path to experiment configuration file")
    parser.add_argument("--quick", action="store_true", 
                       help="Run quick experiment with minimal settings")
    parser.add_argument("--output_dir", help="Output directory for experiment results")
    
    args = parser.parse_args()
    
    if args.quick:
        # Override config for quick testing
        config = {
            "models_to_test": ["bert-base-uncased"],
            "batch_sizes": [32],
            "learning_rates": [5e-5],
            "epochs": [50],
            "run_test_evaluation": True,
            "use_early_stopping": True,
            "early_stopping_patience": 10
        }
        
        config_path = "quick_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        runner = ExperimentRunner(config_path, args.output_dir)
    else:
        runner = ExperimentRunner(args.config, args.output_dir)
    
    try:
        runner.run_full_experiment()
        logger.info(f"✅  experiment completed")

    except KeyboardInterrupt:
        logger.info("⏹️  Experiment interrupted by user")
    except Exception as e:
        logger.error(f"💥 Experiment failed with error: {e}")

if __name__ == "__main__":
    main() 