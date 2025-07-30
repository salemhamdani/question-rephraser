#!/usr/bin/env python3
"""
Question Rephrasing Streamlit App
=================================

A chat-like interface for testing trained question rephrasing models.
Users can select different models and input disfluent questions to get clean, fluent versions.
"""

import streamlit as st
import torch
import time
from pathlib import Path
from typing import Dict, Tuple
from transformers import (
    T5ForConditionalGeneration, T5Tokenizer,
    BartForConditionalGeneration, BartTokenizer,
    AutoModel, AutoTokenizer
    # EncoderDecoderModel, BertTokenizer  # Commented out BERT models
)
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelLoader:
    """Utility class for loading trained models from Hugging Face Hub and local experiments folder"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._available_models = None
        self._local_models = None
        
    def _discover_local_models(self) -> Dict[str, str]:
        """Discover locally trained models from experiments folder"""
        local_models = {}
        experiments_dir = Path("experiments")
        
        if not experiments_dir.exists():
            return local_models
            
        for model_dir in experiments_dir.iterdir():
            if model_dir.is_dir():
                if "facebook" in model_dir.name.lower():
                    # Check for BART models in facebook subdirectory
                    for bart_dir in model_dir.iterdir():
                        if bart_dir.is_dir() and "bart" in bart_dir.name.lower():
                            model_files = list(bart_dir.glob("*.safetensors")) + list(bart_dir.glob("*.bin"))
                            config_file = bart_dir / "config.json"
                            tokenizer_files = list(bart_dir.glob("tokenizer*")) + list(bart_dir.glob("*.json"))
                            
                            if model_files and config_file.exists() and tokenizer_files:
                                local_models["BART Base (Local)"] = str(bart_dir)
                                logger.info(f"Found local BART model at {bart_dir}")
                                
                else:
                    # Check if this directory contains a trained model
                    model_files = list(model_dir.glob("*.safetensors")) + list(model_dir.glob("*.bin"))
                    config_file = model_dir / "config.json"
                    tokenizer_files = list(model_dir.glob("tokenizer*")) + list(model_dir.glob("*.json"))
                
                    if model_files and config_file.exists() and tokenizer_files:
                    # Extract model name from directory name
                        model_name = model_dir.name
                    
                    # Determine model type from directory name or config
                        if "t5" in model_name.lower():
                            display_name = "T5 Small (Local)"
                            model_type = "t5"
                        elif "bert" in model_name.lower():
                            display_name = "BERT (Local)"
                            model_type = "bert"
                        else:
                            display_name = f"Model (Local: {model_name})"
                            model_type = "unknown"
                    
                        local_models[display_name] = str(model_dir)
        
        return local_models
    
    @st.cache_resource
    def get_available_models(_self) -> Dict[str, str]:
        """Get list of available models from Hugging Face Hub, fallback to local experiments folder"""
        if _self._available_models is not None:
            return _self._available_models
            
        # Try Hugging Face models first
        hf_models = {
            "T5 Small": "salman2025/t5-small",
            "BART Base": "salman2025/bart-base",
        }
        
        # Test if Hugging Face models are accessible
        available_models = {}
        for display_name, model_name in hf_models.items():
            try:
                # Quick test to see if model is accessible
                if "t5" in model_name.lower():
                    T5Tokenizer.from_pretrained(model_name, use_fast=False)
                elif "bart" in model_name.lower():
                    BartTokenizer.from_pretrained(model_name, use_fast=False)
                available_models[display_name] = model_name
                logger.info(f"Hugging Face model {model_name} is accessible")
            except Exception as e:
                logger.warning(f"Hugging Face model {model_name} not accessible: {e}")
        
        # If no Hugging Face models are available, try local models
        if not available_models:
            logger.info("No Hugging Face models available, checking local experiments folder...")
            local_models = _self._discover_local_models()
            if local_models:
                available_models = local_models
                logger.info(f"Found {len(local_models)} local models")
            else:
                logger.warning("No local models found in experiments folder")
        
        _self._available_models = available_models
        return available_models
    
    @st.cache_resource
    def load_model(_self, model_name: str) -> Tuple[object, object, str]:
        """Load model and tokenizer from Hugging Face Hub or local directory"""
        try:
            # Check if this is a local model path
            model_path = Path(model_name)
            is_local = model_path.exists() and model_path.is_dir()
            
            if is_local:
                # Load from local directory
                logger.info(f"Loading local model from {model_name}")
                
                # Determine model type from directory name or config
                if "t5" in model_name.lower():
                    tokenizer = T5Tokenizer.from_pretrained(model_name)
                    model = T5ForConditionalGeneration.from_pretrained(model_name)
                    model_type = "t5"
                elif "bart" in model_name.lower():
                    tokenizer = BartTokenizer.from_pretrained(model_name)
                    model = BartForConditionalGeneration.from_pretrained(model_name)
                    model_type = "bart"
                elif "bert" in model_name.lower():
                    # For BERT models, we'll use AutoModel and AutoTokenizer
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                    model = AutoModel.from_pretrained(model_name)
                    model_type = "bert"
                else:
                    # Try to auto-detect model type
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                    model = AutoModel.from_pretrained(model_name)
                    model_type = "auto"
                    
            else:
                # Load from Hugging Face Hub
                logger.info(f"Loading model from Hugging Face Hub: {model_name}")
                
                # Determine model type from model name
                if "t5" in model_name.lower():
                    tokenizer = T5Tokenizer.from_pretrained(model_name)
                    model = T5ForConditionalGeneration.from_pretrained(model_name)
                    model_type = "t5"
                elif "bart" in model_name.lower():
                    tokenizer = BartTokenizer.from_pretrained(model_name)
                    model = BartForConditionalGeneration.from_pretrained(model_name)
                    model_type = "bart"
                else:
                    raise ValueError(f"Unknown model type for {model_name}")
            
            model.to(_self.device)
            model.eval()
            
            source = "local directory" if is_local else "Hugging Face Hub"
            logger.info(f"Loaded fine-tuned {model_type} model from {source}: {model_name}")
            return model, tokenizer, model_type
            
        except Exception as e:
            logger.error(f"Error loading model from {model_name}: {e}")
            st.error(f"Failed to load model: {e}")
            return None, None, None

def rephrase_question(
    question: str, 
    model: object, 
    tokenizer: object, 
    model_type: str,
    max_length: int = 128,
    device: torch.device = None
) -> str:
    """Rephrase a disfluent question using the loaded model"""
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        # Prepare input based on model type
        if model_type == "t5":
            input_text = f"rephrase: {question}"
        else:
            input_text = question
        
        # Tokenize input
        inputs = tokenizer(
            input_text,
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(device)
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                num_beams=4,
                early_stopping=True,
                do_sample=False,
                temperature=0.7,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode output
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Clean up output for T5 models (remove input prefix)
        if model_type == "t5" and generated_text.startswith("rephrase:"):
            generated_text = generated_text[len("rephrase:"):].strip()
        
        return generated_text
        
    except Exception as e:
        logger.error(f"Error during rephrasing: {e}")
        return f"Error: {str(e)}"

def main():
    """Main Streamlit app"""
    
    # Page configuration
    st.set_page_config(
        page_title="Question Rephraser",
        page_icon="��",
        layout="centered"
    )
    
    # Add custom CSS for better styling and centering
    st.markdown("""
    <style>
    .main > div {
        padding-top: 2rem;
        max-width: 1000px;
        margin: 0 auto;
    }
    
    /* Style chat input to be less wide */
    .stChatInput > div {
        max-width: 100%;
        margin: 0;
    }
    
    /* Style chat messages to be less wide */
    .stChatMessage {
        max-width: 100%;
        margin: 0 auto;
    }
    
    /* Center selectbox */
    .stSelectbox > div {
        max-width: 100%;
        margin: 0;
    }
    
    /* Center buttons */
    .stButton > button {
        display: block;
        margin: 0 auto;
    }
    
    /* Center title and headers */
    h1, h2, h3 {
        text-align: center;
    }
    
    /* Center description */
    .main .block-container {
        max-width: 1000px;
        padding-left: 1rem;
        padding-right: 1rem;
        padding-bottom: 150px;
    }
    
    /* Style expander */
    .streamlit-expanderHeader {
        text-align: center;
    }
    
    /* Add some spacing between sections */
    .stHeader {
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    
    /* Left-aligned notification messages */
    .stAlert {
        max-width: 400px;
        margin: 0;
        position: fixed;
        top: 80px;
        right: 20px;
        z-index: 1001;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        border: none;
        padding: 12px 16px;
        font-size: 14px;
        font-weight: 500;
        animation: slideIn 0.3s ease-out;
        padding: 0;
    }
    
    @keyframes slideIn {
        from {
            transform: translateX(-100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    /* Success notification specific styling */
    .stAlert[data-baseweb="notification"] {
        background: linear-gradient(135deg, #4CAF50, #45a049);
        color: white;
        border-left: 4px solid #2E7D32;
    }
    
    /* Fixed bottom input section */
    .fixed-bottom {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: white;
        z-index: 1000;
        padding: 1rem;
        border-top: 2px solid #e0e0e0;
        box-shadow: 0 -2px 10px rgba(0,0,0,0.1);
    }
    
    /* Hide empty labels from selectbox */
    label[data-testid="stWidgetLabel"] {
        display: none !important;
    }
    
    /* Add padding bottom to main block container */
    .stMainBlockContainer {
        padding-bottom: 2rem;
        padding-top: 1.5rem;
    }
                
    .st-key-chat-history-section{
        max-height: 15rem;
        overflow-y: auto;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Title and description
    st.title("🔄 Question Rephraser Chat")
    st.markdown("""
    Transform disfluent questions into clean, well-formed questions using pre-trained BART and T5 models from Hugging Face Hub (with local fallback).
    
    **Examples of disfluent questions:**
    - "What do petrologists no what do unstable isotope studies indicate?"
    - "When a pathogen is met again scratch that I mean long-lived memory cells are capable of remembering previous encounters with what?"
    """)
    
    # Initialize model loader
    if 'model_loader' not in st.session_state:
        st.session_state.model_loader = ModelLoader()
    
    # Get available models
    available_models = st.session_state.model_loader.get_available_models()
    
    # Check if any models are available
    if not available_models:
        st.error("❌ No models available!")
        st.markdown("""
        **No models found!** The app tried to load models from:
        
        1. **Hugging Face Hub**: Models were not accessible
        2. **Local experiments folder**: No trained models found
        
        **To fix this:**
        - Ensure internet connection for Hugging Face models
        - Or train local models using `python model_trainer.py` and save them in the `experiments/` folder
        """)
        return

    # Initialize chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Display chat history first

    if st.session_state.chat_history:
        # Container for chat messages with customizable width
        with st.container(key="chat-history-section"):
            for i, message in enumerate(st.session_state.chat_history):
                if message["role"] == "user":
                    with st.chat_message("user"):
                        st.write("**Disfluent Question:**")
                        st.write(message["content"])
                else:
                    with st.chat_message("assistant"):
                        st.write(f"**Rephrased ({message.get('model', 'Unknown')}):**")
                        st.write(message["content"])
        
        
        # Center the clear history button
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("🗑️ Clear Chat History", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()
    
    # Fixed bottom input section
    st.markdown('<div class="fixed-bottom">', unsafe_allow_html=True)
    
    # Model selection and input section
    col1, col2 = st.columns([1, 2])
    
    with col1:
        selected_model_name = st.selectbox(
            "",
            options=list(available_models.keys()),
            key="model_selectbox"
        )
    
    with col2:
        # Chat input
        user_input = st.chat_input(
            placeholder="Type your question with disfluencies and press Enter...",
            key="chat_input"
        )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Auto-load model when selection changes
    if selected_model_name != st.session_state.get('current_model_name', None):
        # Determine loading source for spinner message
        if "(Local)" in selected_model_name:
            loading_msg = f"Loading {selected_model_name} from local experiments folder..."
        else:
            loading_msg = f"Loading {selected_model_name} from Hugging Face Hub..."
            
        with st.spinner(loading_msg):
            model_name = available_models[selected_model_name]
            model, tokenizer, model_type = st.session_state.model_loader.load_model(model_name)
            
            if model is not None:
                st.session_state.current_model = model
                st.session_state.current_tokenizer = tokenizer
                st.session_state.current_model_type = model_type
                st.session_state.current_model_name = selected_model_name
                st.success(f"✅ {selected_model_name} loaded successfully!", icon="✅")
            else:
                st.error("❌ Failed to load model")
                return
    
    # Process user input when submitted
    if user_input and user_input.strip():
        # Check if model is loaded
        if 'current_model' not in st.session_state:
            st.warning("Please select and load a model first!")
        else:
            # Add user message to chat history
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_input,
                "timestamp": time.time()
            })
            
            # Generate response
            with st.spinner("Rephrasing your question..."):
                rephrased = rephrase_question(
                    user_input,
                    st.session_state.current_model,
                    st.session_state.current_tokenizer,
                    st.session_state.current_model_type,
                    max_length=128  # Fixed at 128 tokens
                )
            
            # Add assistant response to chat history
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": rephrased,
                "model": st.session_state.current_model_name,
                "timestamp": time.time()
            })
            
            # Rerun to update the chat display
            st.rerun()

if __name__ == "__main__":
    main()