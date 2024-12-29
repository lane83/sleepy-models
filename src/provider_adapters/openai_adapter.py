#!/usr/bin/env python3
"""
OpenAI API adapter for Sleepy-Models.
Handles interactions with OpenAI's GPT models.
"""

import os
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import openai
from dataclasses import dataclass
import json
import tiktoken


@dataclass
class OpenAIResponse:
    """Structured response from OpenAI API."""
    text: str
    model: str
    usage: Dict[str, int]
    finish_reason: str
    metadata: Dict[str, Any]


class OpenAIAdapter:
    """Adapter for OpenAI API interactions."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize OpenAI adapter.
        
        Args:
            api_key (Optional[str]): OpenAI API key
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided")
        
        self.client = openai.OpenAI(api_key=self.api_key)
        
        # Model configurations
        self.models = {
            "gpt-4": {
                "max_tokens": 4096,
                "token_limit": 128000,
                "supports_functions": True,
                "supports_vision": False
            },
            "gpt-4-turbo": {
                "max_tokens": 4096,
                "token_limit": 128000,
                "supports_functions": True,
                "supports_vision": True
            },
            "gpt-3.5-turbo": {
                "max_tokens": 4096,
                "token_limit": 16384,
                "supports_functions": True,
                "supports_vision": False
            }
        }
        
        # Initialize tokenizers
        self.tokenizers = {}
        for model in self.models:
            try:
                self.tokenizers[model] = tiktoken.encoding_for_model(model)
            except Exception as e:
                print(f"Error loading tokenizer for {model}: {e}")
        
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

    def format_messages(self, messages: List[Dict]) -> List[Dict]:
        """
        Format messages for OpenAI API.
        
        Args:
            messages (List[Dict]): Input messages
            
        Returns:
            List[Dict]: Formatted messages
        """
        formatted = []
        
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            
            if role == "system":
                formatted.append({
                    "role": "system",
                    "content": content
                })
            elif role == "user":
                formatted.append({
                    "role": "user",
                    "content": content
                })
            elif role == "assistant":
                formatted.append({
                    "role": "assistant",
                    "content": content
                })
            elif role == "function":
                formatted.append({
                    "role": "function",
                    "name": message.get("name", "function"),
                    "content": content
                })
        
        return formatted

    def complete(self,
                messages: List[Dict],
                model: str = "gpt-4",
                **kwargs) -> OpenAIResponse:
        """
        Generate completion using OpenAI API.
        
        Args:
            messages (List[Dict]): Input messages
            model (str): Model name
            **kwargs: Additional parameters
            
        Returns:
            OpenAIResponse: Structured response
        """
        if not self.validate_model(model):
            raise ValueError(f"Unsupported model: {model}")
        
        try:
            # Format messages
            formatted_messages = self.format_messages(messages)
            
            # Get model config
            model_config = self.models[model]
            
            # Handle function calling
            functions = kwargs.get("functions", None)
            function_call = kwargs.get("function_call", None)
            
            if functions and not model_config["supports_functions"]:
                raise ValueError(f"Model {model} does not support function calling")
            
            # Prepare request parameters
            params = {
                "model": model,
                "messages": formatted_messages,
                "max_tokens": kwargs.get("max_tokens", model_config["max_tokens"]),
                "temperature": kwargs.get("temperature", 0.7),
                "top_p": kwargs.get("top_p", 0.9),
                "stop": kwargs.get("stop", None),
                "presence_penalty": kwargs.get("presence_penalty", 0),
                "frequency_penalty": kwargs.get("frequency_penalty", 0),
                "response_format": kwargs.get("response_format", None)
            }
            
            # Add function calling if provided
            if functions:
                params["functions"] = functions
                if function_call:
                    params["function_call"] = function_call
            
            # Make API request
            response = self.client.chat.completions.create(**params)
            
            # Update tracking
            self.last_response_time = datetime.now()
            self.total_requests += 1
            self.total_tokens += response.usage.total_tokens
            
            # Structure response
            return OpenAIResponse(
                text=response.choices[0].message.content,
                model=model,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                },
                finish_reason=response.choices[0].finish_reason,
                metadata={
                    "id": response.id,
                    "created": response.created,
                    "system_fingerprint": response.system_fingerprint,
                    "function_call": (
                        response.choices[0].message.function_call
                        if hasattr(response.choices[0].message, "function_call")
                        else None
                    )
                }
            )
        
        except openai.RateLimitError as e:
            raise RuntimeError(f"Rate limit exceeded: {str(e)}")
        except openai.APIError as e:
            raise RuntimeError(f"OpenAI API error: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Error generating completion: {str(e)}")

    def stream_complete(self,
                       messages: List[Dict],
                       model: str = "gpt-4",
                       **kwargs) -> Any:
        """
        Stream completion using OpenAI API.
        
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
            # Format messages
            formatted_messages = self.format_messages(messages)
            
            # Get model config
            model_config = self.models[model]
            
            # Prepare request parameters
            params = {
                "model": model,
                "messages": formatted_messages,
                "max_tokens": kwargs.get("max_tokens", model_config["max_tokens"]),
                "temperature": kwargs.get("temperature", 0.7),
                "top_p": kwargs.get("top_p", 0.9),
                "stop": kwargs.get("stop", None),
                "stream": True
            }
            
            # Make streaming request
            stream = self.client.chat.completions.create(**params)
            
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
            
            # Update tracking
            self.last_response_time = datetime.now()
            self.total_requests += 1
        
        except openai.RateLimitError as e:
            raise RuntimeError(f"Rate limit exceeded: {str(e)}")
        except openai.APIError as e:
            raise RuntimeError(f"OpenAI API error: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Error streaming completion: {str(e)}")

    def get_token_count(self, text: str, model: str = "gpt-4") -> int:
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
                raise ValueError(f"No tokenizer available for model {model}")
            
            tokenizer = self.tokenizers[model]
            return len(tokenizer.encode(text))
        
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
            )
        }

    def validate_response_format(self,
                               format_type: str,
                               model: str = "gpt-4") -> bool:
        """
        Validate if response format is supported.
        
        Args:
            format_type (str): Response format type
            model (str): Model name
            
        Returns:
            bool: Whether format is supported
        """
        supported_formats = {
            "gpt-4": ["text", "json"],
            "gpt-4-turbo": ["text", "json"],
            "gpt-3.5-turbo": ["text", "json"]
        }
        
        return (model in supported_formats and
                format_type in supported_formats[model])