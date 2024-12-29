#!/usr/bin/env python3
"""
Anthropic API adapter for Sleepy-Models.
Handles interactions with Anthropic's Claude models.
"""

import os
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import anthropic
from dataclasses import dataclass
import json


@dataclass
class AnthropicResponse:
    """Structured response from Anthropic API."""
    text: str
    model: str
    usage: Dict[str, int]
    finish_reason: str
    metadata: Dict[str, Any]


class AnthropicAdapter:
    """Adapter for Anthropic API interactions."""
    
    def __init__(self, api_key: Optional[str] = None, system_monitor=None):
        """
        Initialize Anthropic adapter.
        
        Args:
            api_key (Optional[str]): Anthropic API key
            system_monitor: Reference to system monitor for telemetry
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API key not provided")
        
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.system_monitor = system_monitor
        
        # Model configurations with local processing constraints
        self.models = {
            "claude-3-opus": {
                "max_tokens": 4096,
                "token_limit": 200000,
                "supports_functions": True,
                "supports_vision": True,
                "local_throughput": 10  # bits/second
            },
            "claude-3-sonnet": {
                "max_tokens": 4096,
                "token_limit": 200000,
                "supports_functions": True,
                "supports_vision": True,
                "local_throughput": 10  # bits/second
            }
        }
        
        # Enhanced tracking metadata
        self.last_response_time: Optional[datetime] = None
        self.total_requests: int = 0
        self.total_tokens: int = 0
        self.current_throughput: float = 0.0
        self.response_times: List[float] = []
        self.error_count: int = 0

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
        Format messages for Anthropic API.
        
        Args:
            messages (List[Dict]): Input messages
            
        Returns:
            List[Dict]: Formatted messages
        """
        formatted = []
        
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            
            # Handle system messages
            if role == "system":
                formatted.append({
                    "role": "assistant",
                    "content": f"System instruction: {content}"
                })
            else:
                formatted.append({
                    "role": role,
                    "content": content
                })
        
        return formatted

    def complete(self,
                messages: List[Dict],
                model: str = "claude-3-sonnet",
                **kwargs) -> AnthropicResponse:
        """
        Generate completion using Anthropic API with local processing constraints.
        
        Args:
            messages (List[Dict]): Input messages
            model (str): Model name
            **kwargs: Additional parameters
            
        Returns:
            AnthropicResponse: Structured response
            
        Raises:
            RuntimeError: If system is in tired state or rate limited
        """
        if not self.validate_model(model):
            raise ValueError(f"Unsupported model: {model}")
        
        # Check system state before proceeding
        if self.system_monitor and self.system_monitor.is_tired():
            raise RuntimeError("System is in tired state - initiating dream cycle")
        
        try:
            start_time = datetime.now()
            
            # Format messages with local processing constraints
            formatted_messages = self.format_messages(messages)
            model_config = self.models[model]
            
            # Calculate throughput constraints
            max_tokens = min(
                kwargs.get("max_tokens", model_config["max_tokens"]),
                int(model_config["local_throughput"] * 0.8)  # 80% of capacity
            )
            
            # Prepare request parameters
            params = {
                "model": model,
                "messages": formatted_messages,
                "max_tokens": max_tokens,
                "temperature": kwargs.get("temperature", 0.7),
                "top_p": kwargs.get("top_p", 0.9),
                "stop_sequences": kwargs.get("stop", None)
            }
            
            # Make API request
            response = self.client.messages.create(**params)
            
            # Calculate performance metrics
            end_time = datetime.now()
            response_time = (end_time - start_time).total_seconds()
            self.response_times.append(response_time)
            self.current_throughput = (
                (response.usage.input_tokens + response.usage.output_tokens) /
                response_time
            )
            
            # Update tracking
            self.last_response_time = end_time
            self.total_requests += 1
            self.total_tokens += (
                response.usage.input_tokens +
                response.usage.output_tokens
            )
            
            # Record telemetry
            if self.system_monitor:
                self.system_monitor.record_request({
                    "model": model,
                    "tokens": response.usage.input_tokens + response.usage.output_tokens,
                    "response_time": response_time,
                    "throughput": self.current_throughput,
                    "finish_reason": response.stop_reason
                })
            
            # Structure response
            return AnthropicResponse(
                text=response.content[0].text,
                model=model,
                usage={
                    "prompt_tokens": response.usage.input_tokens,
                    "completion_tokens": response.usage.output_tokens,
                    "total_tokens": (
                        response.usage.input_tokens +
                        response.usage.output_tokens
                    )
                },
                finish_reason=response.stop_reason,
                metadata={
                    "system_fingerprint": response.system_fingerprint,
                    "role": response.role,
                    "response_time": response_time,
                    "throughput": self.current_throughput
                }
            )
        
        except Exception as e:
            self.error_count += 1
            # Record error in telemetry
            if self.system_monitor:
                self.system_monitor.record_error({
                    "model": model,
                    "error_type": str(type(e)),
                    "error_message": str(e),
                    "timestamp": datetime.now().isoformat()
                })
            
            # Handle rate limits and other errors
            if "rate_limit" in str(e).lower():
                raise RuntimeError(f"Rate limit exceeded: {str(e)}")
            raise RuntimeError(f"Anthropic API error: {str(e)}")

    def stream_complete(self,
                       messages: List[Dict],
                       model: str = "claude-3-sonnet",
                       **kwargs) -> Any:
        """
        Stream completion using Anthropic API.
        
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
                "stop_sequences": kwargs.get("stop", None),
                "stream": True
            }
            
            # Make streaming request
            with self.client.messages.stream(**params) as stream:
                for chunk in stream:
                    if chunk.content:
                        yield chunk.content[0].text
            
            # Update tracking
            self.last_response_time = datetime.now()
            self.total_requests += 1
        
        except Exception as e:
            # Handle rate limits and other errors
            if "rate_limit" in str(e).lower():
                raise RuntimeError(f"Rate limit exceeded: {str(e)}")
            raise RuntimeError(f"Anthropic API error: {str(e)}")

    def get_token_count(self, text: str, model: str = "claude-3-sonnet") -> int:
        """
        Get token count for text.
        
        Args:
            text (str): Input text
            model (str): Model name
            
        Returns:
            int: Token count
        """
        try:
            response = self.client.count_tokens(text)
            return response.count
        except Exception as e:
            raise RuntimeError(f"Error counting tokens: {str(e)}")

    def get_usage_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive usage statistics.
        
        Returns:
            Dict[str, Any]: Usage statistics including performance metrics
        """
        avg_response_time = (
            sum(self.response_times) / len(self.response_times)
            if self.response_times else 0
        )
        
        return {
            "total_requests": self.total_requests,
            "total_tokens": self.total_tokens,
            "last_response_time": (
                self.last_response_time.isoformat()
                if self.last_response_time else None
            ),
            "current_throughput": self.current_throughput,
            "average_response_time": avg_response_time,
            "error_count": self.error_count,
            "system_state": (
                self.system_monitor.get_state()
                if self.system_monitor else None
            )
        }
