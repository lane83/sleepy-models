#!/usr/bin/env python3
"""
Model management system for Sleepy-Models.
Handles model initialization, caching, and optimization.
"""

import os
import time
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import threading
import json
from pathlib import Path
import numpy as np
from dataclasses import dataclass
import torch
from transformers import AutoTokenizer, AutoModel
import anthropic
import openai
from dotenv import load_dotenv


@dataclass
class ModelConfig:
    """Configuration for a model."""
    provider: str
    model_name: str
    max_tokens: int
    token_limit: int
    temperature: float
    top_p: float
    cache_size: int
    batch_size: Optional[int]
    device: str  # 'cpu', 'cuda', 'mps'
    quantization: Optional[str]  # '4bit', '8bit', None
    metadata: Dict[str, Any]


@dataclass
class ModelStats:
    """Statistics for model usage."""
    total_requests: int
    total_tokens: int
    average_latency: float
    error_count: int
    last_error: Optional[datetime]
    cache_hits: int
    cache_misses: int
    total_cost: float


class ModelManager:
    """Manages model initialization and operation."""
    
    def __init__(self, 
                 rate_limiter,
                 request_scheduler,
                 usage_tracker,
                 data_dir: str = "data"):
        """
        Initialize the model manager.
        
        Args:
            rate_limiter: RateLimiter instance
            request_scheduler: RequestScheduler instance
            usage_tracker: UsageTracker instance
            data_dir (str): Directory to store model data
        """
        self.usage_tracker = usage_tracker
        # Local processing constraints (10 bits/s)
        self.processing_rate = 10  # bits per second
        self.last_processed_time = datetime.now()
        self.processed_bits = 0
        self.rate_limiter = rate_limiter
        self.request_scheduler = request_scheduler
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Load environment variables
        load_dotenv()
        
        # Initialize provider clients
        self.clients = {
            "anthropic": anthropic.Anthropic(
                api_key=os.getenv("ANTHROPIC_API_KEY")
            ),
            "openai": openai.OpenAI(
                api_key=os.getenv("OPENAI_API_KEY")
            ),
            "huggingface": None  # Initialized per model
        }
        
        # Model configurations
        self.model_configs: Dict[str, ModelConfig] = {}
        
        # Model instances and tokenizers
        self.models: Dict[str, Any] = {}
        self.tokenizers: Dict[str, Any] = {}
        
        # Response cache
        self.response_cache: Dict[str, Dict] = {}
        
        # Model statistics
        self.model_stats: Dict[str, ModelStats] = {}
        
        # Locks
        self.cache_lock = threading.Lock()
        self.stats_lock = threading.Lock()
        
        # Load configurations and stats
        self.load_configs()
        self.load_stats()
        
        # Initialize default models
        self._initialize_default_models()

    def load_configs(self) -> None:
        """Load model configurations."""
        config_file = self.data_dir / "model_configs.json"
        try:
            if config_file.exists():
                with open(config_file, 'r') as f:
                    configs = json.load(f)
                    for model_id, config in configs.items():
                        self.model_configs[model_id] = ModelConfig(**config)
        except Exception as e:
            print(f"Error loading model configs: {e}")

    def save_configs(self) -> None:
        """Save model configurations."""
        config_file = self.data_dir / "model_configs.json"
        try:
            configs = {
                model_id: {
                    "provider": config.provider,
                    "model_name": config.model_name,
                    "max_tokens": config.max_tokens,
                    "token_limit": config.token_limit,
                    "temperature": config.temperature,
                    "top_p": config.top_p,
                    "cache_size": config.cache_size,
                    "batch_size": config.batch_size,
                    "device": config.device,
                    "quantization": config.quantization,
                    "metadata": config.metadata
                }
                for model_id, config in self.model_configs.items()
            }
            
            temp_file = config_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(configs, f)
            temp_file.replace(config_file)
        
        except Exception as e:
            print(f"Error saving model configs: {e}")
            if temp_file.exists():
                temp_file.unlink()

    def load_stats(self) -> None:
        """Load model statistics."""
        stats_file = self.data_dir / "model_stats.json"
        try:
            if stats_file.exists():
                with open(stats_file, 'r') as f:
                    stats = json.load(f)
                    for model_id, stat in stats.items():
                        self.model_stats[model_id] = ModelStats(
                            total_requests=stat["total_requests"],
                            total_tokens=stat["total_tokens"],
                            average_latency=stat["average_latency"],
                            error_count=stat["error_count"],
                            last_error=datetime.fromisoformat(stat["last_error"])
                                if stat["last_error"] else None,
                            cache_hits=stat["cache_hits"],
                            cache_misses=stat["cache_misses"],
                            total_cost=stat["total_cost"]
                        )
        except Exception as e:
            print(f"Error loading model stats: {e}")

    def save_stats(self) -> None:
        """Save model statistics."""
        stats_file = self.data_dir / "model_stats.json"
        try:
            stats = {
                model_id: {
                    "total_requests": stat.total_requests,
                    "total_tokens": stat.total_tokens,
                    "average_latency": stat.average_latency,
                    "error_count": stat.error_count,
                    "last_error": stat.last_error.isoformat()
                        if stat.last_error else None,
                    "cache_hits": stat.cache_hits,
                    "cache_misses": stat.cache_misses,
                    "total_cost": stat.total_cost
                }
                for model_id, stat in self.model_stats.items()
            }
            
            temp_file = stats_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(stats, f)
            temp_file.replace(stats_file)
        
        except Exception as e:
            print(f"Error saving model stats: {e}")
            if temp_file.exists():
                temp_file.unlink()

    def _initialize_default_models(self) -> None:
        """Initialize default model configurations."""
        default_configs = {
            "claude-3-sonnet": ModelConfig(
                provider="anthropic",
                model_name="claude-3-sonnet",
                max_tokens=4096,
                token_limit=200000,
                temperature=0.7,
                top_p=0.9,
                cache_size=1000,
                batch_size=None,
                device="cpu",
                quantization=None,
                metadata={"type": "chat"}
            ),
            "claude-3-opus": ModelConfig(
                provider="anthropic",
                model_name="claude-3-opus",
                max_tokens=4096,
                token_limit=200000,
                temperature=0.7,
                top_p=0.9,
                cache_size=1000,
                batch_size=None,
                device="cpu",
                quantization=None,
                metadata={"type": "chat"}
            ),
            "gpt-4": ModelConfig(
                provider="openai",
                model_name="gpt-4",
                max_tokens=4096,
                token_limit=128000,
                temperature=0.7,
                top_p=0.9,
                cache_size=1000,
                batch_size=None,
                device="cpu",
                quantization=None,
                metadata={"type": "chat"}
            ),
            "phi-2": ModelConfig(
                provider="huggingface",
                model_name="microsoft/phi-2",
                max_tokens=2048,
                token_limit=2048,
                temperature=0.7,
                top_p=0.9,
                cache_size=1000,
                batch_size=4,
                device="cuda" if torch.cuda.is_available() else "cpu",
                quantization="4bit",
                metadata={"type": "completion"}
            )
        }
        
        # Add default configs if not already present
        for model_id, config in default_configs.items():
            if model_id not in self.model_configs:
                self.model_configs[model_id] = config
                self.model_stats[model_id] = ModelStats(
                    total_requests=0,
                    total_tokens=0,
                    average_latency=0.0,
                    error_count=0,
                    last_error=None,
                    cache_hits=0,
                    cache_misses=0,
                    total_cost=0.0
                )

    def add_model(self, model_id: str, config: ModelConfig) -> bool:
        """
        Add a new model configuration.
        
        Args:
            model_id (str): Model identifier
            config (ModelConfig): Model configuration
            
        Returns:
            bool: Success status
        """
        try:
            self.model_configs[model_id] = config
            self.model_stats[model_id] = ModelStats(
                total_requests=0,
                total_tokens=0,
                average_latency=0.0,
                error_count=0,
                last_error=None,
                cache_hits=0,
                cache_misses=0,
                total_cost=0.0
            )
            
            self.save_configs()
            self.save_stats()
            return True
        
        except Exception as e:
            print(f"Error adding model: {e}")
            return False

    def remove_model(self, model_id: str) -> bool:
        """
        Remove a model configuration.
        
        Args:
            model_id (str): Model identifier
            
        Returns:
            bool: Success status
        """
        try:
            if model_id in self.model_configs:
                del self.model_configs[model_id]
                del self.model_stats[model_id]
                
                if model_id in self.models:
                    del self.models[model_id]
                if model_id in self.tokenizers:
                    del self.tokenizers[model_id]
                if model_id in self.response_cache:
                    del self.response_cache[model_id]
                
                self.save_configs()
                self.save_stats()
                return True
            return False
        
        except Exception as e:
            print(f"Error removing model: {e}")
            return False

    def get_model(self, model_id: str) -> Tuple[Optional[Any], Optional[Any]]:
        """
        Get model and tokenizer instances.
        
        Args:
            model_id (str): Model identifier
            
        Returns:
            Tuple[Optional[Any], Optional[Any]]: (model, tokenizer) instances
        """
        try:
            if model_id not in self.model_configs:
                return None, None
            
            config = self.model_configs[model_id]
            
            # Initialize model if not already loaded
            if model_id not in self.models:
                if config.provider == "huggingface":
                    self.models[model_id] = AutoModel.from_pretrained(
                        config.model_name,
                        device_map=config.device,
                        load_in_4bit=config.quantization == "4bit",
                        load_in_8bit=config.quantization == "8bit"
                    )
                    self.tokenizers[model_id] = AutoTokenizer.from_pretrained(
                        config.model_name
                    )
            
            return self.models.get(model_id), self.tokenizers.get(model_id)
        
        except Exception as e:
            print(f"Error getting model: {e}")
            return None, None

    def process_message(self,
                       model_id: str,
                       messages: List[Dict],
                       **kwargs) -> Optional[str]:
        """
        Process a message using specified model.
        
        Args:
            model_id (str): Model identifier
            messages (List[Dict]): Message history
            **kwargs: Additional model parameters
            
        Returns:
            Optional[str]: Model response
        """
        try:
            config = self.model_configs.get(model_id)
            if not config:
                raise ValueError(f"Model {model_id} not found")
            
            # Enforce processing rate limit
            now = datetime.now()
            elapsed = (now - self.last_processed_time).total_seconds()
            available_bits = elapsed * self.processing_rate
            
            # Calculate message size in bits
            message_size = sum(len(json.dumps(m)) * 8 for m in messages)
            
            # Wait if we've exceeded the processing rate
            if message_size > available_bits:
                wait_time = (message_size - available_bits) / self.processing_rate
                time.sleep(wait_time)
                self.last_processed_time = datetime.now()
                self.processed_bits = 0
            else:
                self.processed_bits += message_size
                self.last_processed_time = now
            
            # Check cache
            cache_key = self._generate_cache_key(model_id, messages, kwargs)
            cached_response = self._check_cache(model_id, cache_key)
            if cached_response:
                return cached_response
            
            # Schedule request
            response = self._schedule_model_request(
                model_id,
                messages,
                **kwargs
            )
            
            # Update cache
            if response:
                self._update_cache(model_id, cache_key, response)
            
            # Monitor usage and trigger dream state if needed
            self.monitor_usage()
            
            return response
        
        except Exception as e:
            print(f"Error processing message: {e}")
            return None

    def _schedule_model_request(self,
                              model_id: str,
                              messages: List[Dict],
                              **kwargs) -> Optional[str]:
        """
        Schedule a model request.
        
        Args:
            model_id (str): Model identifier
            messages (List[Dict]): Message history
            **kwargs: Additional model parameters
            
        Returns:
            Optional[str]: Model response
        """
        try:
            config = self.model_configs[model_id]
            
            # Prepare request function
            if config.provider == "anthropic":
                request_func = self._anthropic_request
            elif config.provider == "openai":
                request_func = self._openai_request
            elif config.provider == "huggingface":
                request_func = self._huggingface_request
            else:
                raise ValueError(f"Unknown provider: {config.provider}")
            
            # Schedule request
            request_id = self.request_scheduler.schedule_request(
                provider=config.provider,
                model=config.model_name,
                function=request_func,
                args=(model_id, messages),
                kwargs=kwargs
            )
            
            # Wait for response
            while True:
                status = self.request_scheduler.get_request_status(request_id)
                if not status:
                    break
                if status["status"] == "complete":
                    return status["response"]
            
            return None
        
        except Exception as e:
            print(f"Error scheduling model request: {e}")
            return None

    def _anthropic_request(self,
                          model_id: str,
                          messages: List[Dict],
                          **kwargs) -> str:
        """
        Make a request to Anthropic API.
        
        Args:
            model_id (str): Model identifier
            messages (List[Dict]): Message history
            **kwargs: Additional model parameters
            
        Returns:
            str: Model response
        """
        try:
            config = self.model_configs[model_id]
            
            response = self.clients["anthropic"].messages.create(
                model=config.model_name,
                messages=messages,
                max_tokens=kwargs.get("max_tokens", config.max_tokens),
                temperature=kwargs.get("temperature", config.temperature),
                top_p=kwargs.get("top_p", config.top_p)
            )
            
            return response.content[0].text
        
        except Exception as e:
            print(f"Error in Anthropic request: {e}")
            self._update_error_stats(model_id)
            raise

    def _openai_request(self,
                       model_id: str,
                       messages: List[Dict],
                       **kwargs) -> str:
        """
        Make a request to OpenAI API.
        
        Args:
            model_id (str): Model identifier
            messages (List[Dict]): Message history
            **kwargs: Additional model parameters
            
        Returns:
            str: Model response
        """
        try:
            config = self.model_configs[model_id]
            
            response = self.clients["openai"].chat.completions.create(
                model=config.model_name,
                messages=messages,
                max_tokens=kwargs.get("max_tokens", config.max_tokens),
                temperature=kwargs.get("temperature", config.temperature),
                top_p=kwargs.get("top_p", config.top_p)
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            print(f"Error in OpenAI request: {e}")
            self._update_error_stats(model_id)
            raise

    def _huggingface_request(self,
                            model_id: str,
                            messages: List[Dict],
                            **kwargs) -> str:
        """
        Make a request to local Hugging Face model.
        
        Args:
            model_id (str): Model identifier
            messages (List[Dict]): Message history
            **kwargs: Additional model parameters
            
        Returns:
            str: Model response
        """
        try:
            model, tokenizer = self.get_model(model_id)
            if not model or not tokenizer:
                raise ValueError(f"Model {model_id} not initialized")
            
            config = self.model_configs[model_id]
            
            # Convert messages to prompt
            prompt = self._messages_to_prompt(messages)
            
            # Tokenize input
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=config.token_limit
            ).to(config.device)
            
            # Generate response
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=kwargs.get("max_tokens", config.max_tokens),
                    temperature=kwargs.get("temperature", config.temperature),
                    top_p=kwargs.get("top_p", config.top_p),
                    pad_token_id=tokenizer.pad_token_id,
                    batch_size=config.batch_size
                )
            
            # Decode response
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response
        
        except Exception as e:
            print(f"Error in Hugging Face request: {e}")
            self._update_error_stats(model_id)
            raise

    def _messages_to_prompt(self, messages: List[Dict]) -> str:
        """
        Convert message history to prompt string.
        
        Args:
            messages (List[Dict]): Message history
            
        Returns:
            str: Formatted prompt
        """
        prompt = ""
        for message in messages:
            role = message["role"]
            content = message["content"]
            
            if role == "system":
                prompt += f"System: {content}\n\n"
            elif role == "user":
                prompt += f"Human: {content}\n\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n\n"
        
        return prompt.strip()

    def _generate_cache_key(self,
                           model_id: str,
                           messages: List[Dict],
                           kwargs: Dict) -> str:
        """
        Generate cache key for request.
        
        Args:
            model_id (str): Model identifier
            messages (List[Dict]): Message history
            kwargs (Dict): Additional parameters
            
        Returns:
            str: Cache key
        """
        # Convert messages to stable string representation
        message_str = json.dumps(messages, sort_keys=True)
        kwargs_str = json.dumps(kwargs, sort_keys=True)
        
        # Combine components
        key_components = [model_id, message_str, kwargs_str]
        return "|".join(key_components)

    def _check_cache(self, model_id: str, cache_key: str) -> Optional[str]:
        """
        Check cache for response.
        
        Args:
            model_id (str): Model identifier
            cache_key (str): Cache key
            
        Returns:
            Optional[str]: Cached response if found
        """
        with self.cache_lock:
            if (model_id in self.response_cache and
                cache_key in self.response_cache[model_id]):
                self._update_cache_stats(model_id, True)
                return self.response_cache[model_id][cache_key]
            
            self._update_cache_stats(model_id, False)
            return None

    def _update_cache(self, model_id: str, cache_key: str, response: str) -> None:
        """
        Update response cache.
        
        Args:
            model_id (str): Model identifier
            cache_key (str): Cache key
            response (str): Model response
        """
        with self.cache_lock:
            if model_id not in self.response_cache:
                self.response_cache[model_id] = {}
            
            # Check cache size limit
            config = self.model_configs[model_id]
            if len(self.response_cache[model_id]) >= config.cache_size:
                # Remove oldest entry
                oldest_key = next(iter(self.response_cache[model_id]))
                del self.response_cache[model_id][oldest_key]
            
            self.response_cache[model_id][cache_key] = response

    def _update_cache_stats(self, model_id: str, hit: bool) -> None:
        """
        Update cache statistics.
        
        Args:
            model_id (str): Model identifier
            hit (bool): Whether cache hit occurred
        """
        with self.stats_lock:
            if model_id in self.model_stats:
                if hit:
                    self.model_stats[model_id].cache_hits += 1
                else:
                    self.model_stats[model_id].cache_misses += 1
                self.save_stats()

    def _update_error_stats(self, model_id: str) -> None:
        """
        Update error statistics.
        
        Args:
            model_id (str): Model identifier
        """
        with self.stats_lock:
            if model_id in self.model_stats:
                stats = self.model_stats[model_id]
                stats.error_count += 1
                stats.last_error = datetime.now()
                self.save_stats()

    def get_model_status(self, model_id: str) -> Optional[Dict]:
        """
        Get current status of a model.
        
        Args:
            model_id (str): Model identifier
            
        Returns:
            Optional[Dict]: Model status information
        """
        try:
            if model_id not in self.model_configs:
                return None
            
            config = self.model_configs[model_id]
            stats = self.model_stats[model_id]
            
            return {
                "provider": config.provider,
                "model_name": config.model_name,
                "device": config.device,
                "total_requests": stats.total_requests,
                "total_tokens": stats.total_tokens,
                "average_latency": stats.average_latency,
                "error_rate": stats.error_count / max(stats.total_requests, 1),
                "cache_hit_rate": (stats.cache_hits /
                                 max(stats.cache_hits + stats.cache_misses, 1)),
                "total_cost": stats.total_cost,
                "last_error": stats.last_error.isoformat() if stats.last_error else None
            }
        
        except Exception as e:
            print(f"Error getting model status: {e}")
            return None

    def optimize_models(self) -> None:
        """Optimize model configurations and resources based on collected metrics."""
        try:
            for model_id, config in self.model_configs.items():
                stats = self.model_stats.get(model_id)
                if not stats:
                    continue
                
                # Get usage tracker data for this model
                usage_data = self._get_usage_data(model_id)
                
                # Error rate optimization
                error_rate = stats.error_count / max(stats.total_requests, 1)
                if error_rate > 0.1:  # 10% error rate threshold
                    print(f"High error rate for {model_id}: {error_rate:.2%}")
                    
                    # Adjust batch size for local models
                    if (config.provider == "huggingface" and
                        config.batch_size and
                        config.batch_size > 1):
                        config.batch_size -= 1
                        print(f"Reduced batch size to {config.batch_size}")
                
                # Cache optimization
                hit_rate = (stats.cache_hits /
                          max(stats.cache_hits + stats.cache_misses, 1))
                if hit_rate > 0.8 and config.cache_size < 10000:
                    config.cache_size *= 2
                    print(f"Increased cache size to {config.cache_size}")
                elif hit_rate < 0.2 and config.cache_size > 100:
                    config.cache_size //= 2
                    print(f"Decreased cache size to {config.cache_size}")
                
                # Latency optimization
                if usage_data and usage_data.get("average_latency", 0) > 2.0:  # seconds
                    print(f"High latency for {model_id}: {usage_data['average_latency']:.2f}s")
                    if config.provider == "huggingface":
                        # Reduce model precision for faster inference
                        if config.quantization != "4bit":
                            config.quantization = "4bit"
                            print(f"Set quantization to 4bit for {model_id}")
                
                # Token usage optimization
                if usage_data and usage_data.get("average_tokens", 0) > config.token_limit * 0.8:
                    print(f"High token usage for {model_id}: {usage_data['average_tokens']}")
                    config.max_tokens = min(config.max_tokens * 0.9, config.token_limit)
                    print(f"Reduced max_tokens to {config.max_tokens}")
                
                # Knowledge access optimization
                if usage_data and usage_data.get("knowledge_access_rate", 0) > 0.5:
                    print(f"High knowledge access rate for {model_id}: {usage_data['knowledge_access_rate']:.2f}")
                    if config.provider == "huggingface":
                        # Increase context window for better knowledge integration
                        config.token_limit = min(config.token_limit * 1.1, 8192)
                        print(f"Increased token_limit to {config.token_limit}")
                
                # Uncertainty optimization
                if usage_data and usage_data.get("average_uncertainty", 0) > 0.7:
                    print(f"High uncertainty for {model_id}: {usage_data['average_uncertainty']:.2f}")
                    config.temperature = max(config.temperature * 0.9, 0.1)
                    print(f"Reduced temperature to {config.temperature}")
                
                # Coherence optimization
                if usage_data and usage_data.get("average_coherence", 0) < 0.7:
                    print(f"Low coherence for {model_id}: {usage_data['average_coherence']:.2f}")
                    config.top_p = min(config.top_p * 1.1, 1.0)
                    print(f"Increased top_p to {config.top_p}")
            
            self.save_configs()
        
        except Exception as e:
            print(f"Error optimizing models: {e}")

    def _get_usage_data(self, model_id: str) -> Optional[Dict]:
        """
        Get usage data for a specific model from the usage tracker.
        
        Args:
            model_id (str): Model identifier
            
        Returns:
            Optional[Dict]: Usage data dictionary
        """
        try:
            if not hasattr(self, 'usage_tracker'):
                return None
                
            # Get recent sessions for this model
            cutoff_date = datetime.now() - timedelta(days=7)
            sessions = [
                s for s in self.usage_tracker.history["sessions"]
                if s["model"] == model_id and 
                datetime.fromisoformat(s["start_time"]) > cutoff_date
            ]
            
            if not sessions:
                return None
                
            # Calculate average metrics
            total_tokens = sum(s["input_tokens"] + s["output_tokens"] for s in sessions)
            total_latency = sum(s["latency_breakdown"]["total"] for s in sessions)
            total_knowledge_access = sum(s["knowledge_access_count"] for s in sessions)
            total_uncertainty = sum(sum(s["uncertainty_levels"]) for s in sessions)
            total_coherence = sum(sum(s["coherence_scores"]) for s in sessions)
            
            num_requests = sum(s["requests"] for s in sessions)
            num_uncertainty = sum(len(s["uncertainty_levels"]) for s in sessions)
            num_coherence = sum(len(s["coherence_scores"]) for s in sessions)
            
            return {
                "average_latency": total_latency / len(sessions),
                "average_tokens": total_tokens / num_requests,
                "knowledge_access_rate": total_knowledge_access / num_requests,
                "average_uncertainty": total_uncertainty / num_uncertainty if num_uncertainty > 0 else 0.5,
                "average_coherence": total_coherence / num_coherence if num_coherence > 0 else 0.8
            }
        except Exception as e:
            print(f"Error getting usage data: {e}")
            return None

    def clear_cache(self, model_id: Optional[str] = None) -> None:
        """
        Clear response cache.
        
        Args:
            model_id (str, optional): Model identifier to clear cache for
        """
        with self.cache_lock:
            if model_id:
                if model_id in self.response_cache:
                    del self.response_cache[model_id]
            else:
                self.response_cache.clear()

    def get_total_tokens(self) -> int:
        """
        Get total tokens used across all models.
        
        Returns:
            int: Total token count
        """
        return sum(stats.total_tokens for stats in self.model_stats.values())

    def get_error_rate(self) -> float:
        """
        Get overall error rate across all models.
        
        Returns:
            float: Error rate
        """
        total_requests = sum(stats.total_requests for stats in self.model_stats.values())
        total_errors = sum(stats.error_count for stats in self.model_stats.values())
        return total_errors / max(total_requests, 1)

    def get_token_limit(self) -> int:
        """
        Get minimum token limit across active models.
        
        Returns:
            int: Token limit
        """
        return min(config.token_limit for config in self.model_configs.values())

    def save_context(self) -> Dict[str, Any]:
        """
        Save current context for dream state processing.
        
        Returns:
            Dict[str, Any]: Saved context including models, cache, and stats
        """
        return {
            "models": {k: v.state_dict() for k, v in self.models.items()},
            "cache": self.response_cache,
            "stats": self.model_stats
        }

    def restore_context(self, context: Dict[str, Any]) -> None:
        """
        Restore context after dream state processing.
        
        Args:
            context (Dict[str, Any]): Saved context to restore
        """
        for model_id, state in context["models"].items():
            if model_id in self.models:
                self.models[model_id].load_state_dict(state)
        self.response_cache = context["cache"]
        self.model_stats = context["stats"]

    def trigger_dream_state(self) -> None:
        """
        Initiate dream state processing.
        """
        # Save current context
        context = self.save_context()
        
        # Clear working memory
        self.clear_cache()
        
        # Process memories (this would interface with memory_manager)
        # For now just log the event
        print("Entering dream state...")
        
        # Restore context after processing
        self.restore_context(context)
        print("Exited dream state")

    def check_tiredness(self) -> bool:
        """
        Check if system needs to enter dream state based on usage patterns.
        
        Returns:
            bool: True if dream state should be triggered
        """
        # Calculate total usage metrics
        total_requests = sum(s.total_requests for s in self.model_stats.values())
        total_tokens = sum(s.total_tokens for s in self.model_stats.values())
        error_rate = self.get_error_rate()
        
        # Adaptive thresholds
        request_threshold = 1000 + (total_requests // 1000) * 500
        token_threshold = 100000 + (total_tokens // 100000) * 50000
        error_threshold = 0.1 + (total_requests // 1000) * 0.01
        
        # Check if any threshold is exceeded
        if (total_requests > request_threshold or
            total_tokens > token_threshold or
            error_rate > error_threshold):
            return True
        return False

    def monitor_usage(self) -> None:
        """
        Monitor usage patterns and trigger dream state when needed.
        """
        if self.check_tiredness():
            self.trigger_dream_state()
            # Reset statistics after dream state
            for stats in self.model_stats.values():
                stats.total_requests = 0
                stats.total_tokens = 0
                stats.error_count = 0
