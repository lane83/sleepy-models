#!/usr/bin/env python3
"""
Hugging Face adapter for Sleepy-Models.
Handles interactions with local and API-based Hugging Face models.
"""

import os
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2Seq,
    pipeline,
    TextIteratorStreamer
)
from dataclasses import dataclass
import json
import threading
from queue import Queue
import numpy as np


@dataclass
class HuggingFaceResponse:
    """Structured response from Hugging Face model."""
    text: str
    model: str
    usage: Dict[str, int]
    finish_reason: str
    metadata: Dict[str, Any]


class HuggingFaceAdapter:
    """Adapter for Hugging Face model interactions."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Hugging Face adapter.
        
        Args:
            api_key (Optional[str]): Hugging Face API key
        """
        self.api_key = api_key or os.getenv("HUGGINGFACE_API_KEY")
        
        # Model configurations
        self.models = {
            "microsoft/phi-2": {
                "type": "causal",
                "max_tokens": 2048,
                "token_limit": 2048,
                "supports_chat": True,
                "quantization": "4bit"
            },
            "google/flan-t5-large": {
                "type": "seq2seq",
                "max_tokens": 1024,
                "token_limit": 1024,
                "supports_chat": False,
                "quantization": "8bit"
            },
            "facebook/opt-6.7b": {
                "type": "causal",
                "max_tokens": 2048,
                "token_limit": 2048,
                "supports_chat": True,
                "quantization": "4bit"
            }
        }
        
        # Model and tokenizer instances
        self.loaded_models: Dict[str, Any] = {}
        self.tokenizers: Dict[str, Any] = {}
        
        # Determine device
        self.device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        
        # Tracking metadata
        self.last_response_time: Optional[datetime] = None
        self.total_requests: int = 0
        self.total_tokens: int = 0

    def validate_model(self, model: str) -> bool:
        """
        Validate if model is supported.
        
        Args:
            model (str): Model name
            
        Returns:
            bool: Whether model is supported
        """
        return model in self.models

    def load_model(self, model: str) -> Tuple[Any, Any]:
        """
        Load model and tokenizer.
        
        Args:
            model (str): Model name
            
        Returns:
            Tuple[Any, Any]: (model, tokenizer) instances
        """
        if not self.validate_model(model):
            raise ValueError(f"Unsupported model: {model}")
        
        try:
            # Return if already loaded
            if model in self.loaded_models:
                return self.loaded_models[model], self.tokenizers[model]
            
            config = self.models[model]
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model)
            if not tokenizer.pad_token:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Load model with quantization if specified
            model_args = {
                "device_map": self.device,
                "trust_remote_code": True
            }
            
            if config["quantization"] == "4bit":
                model_args.update({
                    "load_in_4bit": True,
                    "bnb_4bit_compute_dtype": torch.float16
                })
            elif config["quantization"] == "8bit":
                model_args.update({
                    "load_in_8bit": True
                })
            
            # Load appropriate model class
            if config["type"] == "causal":
                model_instance = AutoModelForCausalLM.from_pretrained(
                    model,
                    **model_args
                )
            elif config["type"] == "seq2seq":
                model_instance = AutoModelForSeq2Seq.from_pretrained(
                    model,
                    **model_args
                )
            else:
                raise ValueError(f"Unsupported model type: {config['type']}")
            
            # Store instances
            self.loaded_models[model] = model_instance
            self.tokenizers[model] = tokenizer
            
            return model_instance, tokenizer
        
        except Exception as e:
            raise RuntimeError(f"Error loading model: {str(e)}")

    def format_prompt(self, messages: List[Dict], model: str) -> str:
        """
        Format messages into model-appropriate prompt.
        
        Args:
            messages (List[Dict]): Input messages
            model (str): Model name
            
        Returns:
            str: Formatted prompt
        """
        config = self.models[model]
        prompt = ""
        
        if config["supports_chat"]:
            # Chat-optimized format
            for message in messages:
                role = message.get("role", "user")
                content = message.get("content", "")
                
                if role == "system":
                    prompt += f"System: {content}\n\n"
                elif role == "user":
                    prompt += f"Human: {content}\n\n"
                elif role == "assistant":
                    prompt += f"Assistant: {content}\n\n"
            
            prompt += "Assistant: "
        else:
            # Simple concatenation for non-chat models
            prompt = " ".join(msg["content"] for msg in messages)
        
        return prompt

    def complete(self,
                messages: List[Dict],
                model: str = "microsoft/phi-2",
                **kwargs) -> HuggingFaceResponse:
        """
        Generate completion using Hugging Face model.
        
        Args:
            messages (List[Dict]): Input messages
            model (str): Model name
            **kwargs: Additional parameters
            
        Returns:
            HuggingFaceResponse: Structured response
        """
        if not self.validate_model(model):
            raise ValueError(f"Unsupported model: {model}")
        
        try:
            # Load model and tokenizer
            model_instance, tokenizer = self.load_model(model)
            config = self.models[model]
            
            # Format prompt
            prompt = self.format_prompt(messages, model)
            
            # Tokenize input
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=config["token_limit"]
            ).to(self.device)
            
            # Set generation parameters
            gen_kwargs = {
                "max_new_tokens": kwargs.get("max_tokens", config["max_tokens"]),
                "temperature": kwargs.get("temperature", 0.7),
                "top_p": kwargs.get("top_p", 0.9),
                "pad_token_id": tokenizer.pad_token_id,
                "do_sample": True
            }
            
            # Add stopping criteria if provided
            if "stop" in kwargs:
                stop_token_ids = [
                    tokenizer.encode(stop)[0]
                    for stop in kwargs["stop"]
                ]
                gen_kwargs["stopping_criteria"] = stop_token_ids
            
            # Generate response
            start_time = datetime.now()
            with torch.no_grad():
                outputs = model_instance.generate(
                    **inputs,
                    **gen_kwargs
                )
            
            # Decode response
            response_text = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )
            
            # Calculate token usage
            input_tokens = len(inputs["input_ids"][0])
            output_tokens = len(outputs[0]) - input_tokens
            total_tokens = input_tokens + output_tokens
            
            # Update tracking
            self.last_response_time = datetime.now()
            self.total_requests += 1
            self.total_tokens += total_tokens
            
            # Determine finish reason
            finish_reason = (
                "length" if output_tokens >= gen_kwargs["max_new_tokens"]
                else "stop" if any(stop in response_text for stop in kwargs.get("stop", []))
                else "complete"
            )
            
            # Structure response
            return HuggingFaceResponse(
                text=response_text,
                model=model,
                usage={
                    "prompt_tokens": input_tokens,
                    "completion_tokens": output_tokens,
                    "total_tokens": total_tokens
                },
                finish_reason=finish_reason,
                metadata={
                    "device": str(self.device),
                    "quantization": config["quantization"],
                    "generation_time": (
                        datetime.now() - start_time
                    ).total_seconds()
                }
            )
        
        except Exception as e:
            raise RuntimeError(f"Error generating completion: {str(e)}")

    def stream_complete(self,
                       messages: List[Dict],
                       model: str = "microsoft/phi-2",
                       **kwargs) -> Any:
        """
        Stream completion using Hugging Face model.
        
        Args:
            messages (List[Dict]): Input messages
            model (str): Model name
            **kwargs: Additional parameters
            
        Returns:
            Iterator[str]: Response stream
        """
        if not self.validate_model(model):
            raise ValueError(f"Unsupported model: {model}")
        
        try:
            # Load model and tokenizer
            model_instance, tokenizer = self.load_model(model)
            config = self.models[model]
            
            # Format prompt
            prompt = self.format_prompt(messages, model)
            
            # Tokenize input
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=config["token_limit"]
            ).to(self.device)
            
            # Set up streamer
            streamer = TextIteratorStreamer(
                tokenizer,
                skip_special_tokens=True
            )
            
            # Set generation parameters
            gen_kwargs = {
                "max_new_tokens": kwargs.get("max_tokens", config["max_tokens"]),
                "temperature": kwargs.get("temperature", 0.7),
                "top_p": kwargs.get("top_p", 0.9),
                "pad_token_id": tokenizer.pad_token_id,
                "do_sample": True,
                "streamer": streamer
            }
            
            # Start generation in separate thread
            generation_thread = threading.Thread(
                target=self._generate_with_streamer,
                args=(model_instance, inputs, gen_kwargs)
            )
            generation_thread.start()
            
            # Stream tokens
            for text in streamer:
                yield text
            
            # Update tracking
            self.last_response_time = datetime.now()
            self.total_requests += 1
        
        except Exception as e:
            raise RuntimeError(f"Error streaming completion: {str(e)}")

    def _generate_with_streamer(self,
                              model: Any,
                              inputs: Dict[str, torch.Tensor],
                              gen_kwargs: Dict[str, Any]) -> None:
        """
        Generate text with streamer in separate thread.
        
        Args:
            model (Any): Model instance
            inputs (Dict[str, torch.Tensor]): Input tensors
            gen_kwargs (Dict[str, Any]): Generation parameters
        """
        try:
            with torch.no_grad():
                model.generate(
                    **inputs,
                    **gen_kwargs
                )
        except Exception as e:
            print(f"Error in generation thread: {str(e)}")

    def get_embeddings(self,
                      texts: Union[str, List[str]],
                      model: str = "microsoft/phi-2") -> np.ndarray:
        """
        Get text embeddings using model.
        
        Args:
            texts (Union[str, List[str]]): Input text(s)
            model (str): Model name
            
        Returns:
            np.ndarray: Text embeddings
        """
        if not self.validate_model(model):
            raise ValueError(f"Unsupported model: {model}")
        
        try:
            # Load model and tokenizer
            model_instance, tokenizer = self.load_model(model)
            
            # Ensure list input
            if isinstance(texts, str):
                texts = [texts]
            
            # Get embeddings
            embeddings = []
            
            for text in texts:
                # Tokenize
                inputs = tokenizer(
                    text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                ).to(self.device)
                
                # Get model outputs
                with torch.no_grad():
                    outputs = model_instance(**inputs)
                
                # Use pooled output or mean of last hidden state
                if hasattr(outputs, "pooler_output"):
                    embedding = outputs.pooler_output
                else:
                    embedding = outputs.last_hidden_state.mean(dim=1)
                
                embeddings.append(embedding.cpu().numpy())
            
            return np.vstack(embeddings)
        
        except Exception as e:
            raise RuntimeError(f"Error generating embeddings: {str(e)}")

    def get_token_count(self, text: str, model: str = "microsoft/phi-2") -> int:
        """
        Get token count for text.
        
        Args:
            text (str): Input text
            model (str): Model name
            
        Returns:
            int: Token count
        """
        try:
            if model not in self.tokenizers:
                _, _ = self.load_model(model)
            
            return len(self.tokenizers[model].encode(text))
        
        except Exception as e:
            raise RuntimeError(f"Error counting tokens: {str(e)}")

    def get_usage_stats(self) -> Dict[str, Any]:
        """
        Get usage statistics.
        
        Returns:
            Dict[str, Any]: Usage statistics
        """
        return {
            "total_requests": self.total_requests,
            "total_tokens": self.total_tokens,
            "last_response_time": (
                self.last_response_time.isoformat()
                if self.last_response_time else None
            ),
            "loaded_models": list(self.loaded_models.keys()),
            "device": str(self.device)
        }

    def clear_models(self) -> None:
        """Clear loaded models to free memory."""
        self.loaded_models.clear()
        self.tokenizers.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()