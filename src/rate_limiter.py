#!/usr/bin/env python3
"""
Rate limiting system for Sleepy-Models.
Manages API request rates and quotas for different providers.
"""

import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import threading
from dataclasses import dataclass
import json
from pathlib import Path
import queue
from collections import defaultdict


@dataclass
class RateLimit:
    """Rate limit configuration for an API endpoint."""
    requests_per_minute: int
    requests_per_hour: int
    requests_per_day: int
    concurrent_requests: int
    retry_after: int  # seconds to wait after hitting limit
    cost_per_request: float


@dataclass
class RequestCounter:
    """Tracks request counts for different time windows."""
    minute_count: int
    hour_count: int
    day_count: int
    minute_start: datetime
    hour_start: datetime
    day_start: datetime
    concurrent: int


class RateLimiter:
    """Manages API rate limiting and request queuing."""
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the rate limiter.
        
        Args:
            data_dir (str): Directory to store rate limit data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._last_save_time = datetime.now()
        self._save_interval = timedelta(seconds=30)  # Throttle state saving
        
        # Rate limits for different providers
        self.rate_limits = {
            "anthropic": {
                "default": RateLimit(
                    requests_per_minute=60,
                    requests_per_hour=3000,
                    requests_per_day=50000,
                    concurrent_requests=50,
                    retry_after=60,
                    cost_per_request=0.01
                ),
                "claude-3-sonnet": RateLimit(
                    requests_per_minute=40,
                    requests_per_hour=2000,
                    requests_per_day=30000,
                    concurrent_requests=30,
                    retry_after=60,
                    cost_per_request=0.015
                ),
                "claude-3-opus": RateLimit(
                    requests_per_minute=20,
                    requests_per_hour=1000,
                    requests_per_day=15000,
                    concurrent_requests=20,
                    retry_after=60,
                    cost_per_request=0.03
                )
            },
            "openai": {
                "default": RateLimit(
                    requests_per_minute=60,
                    requests_per_hour=3000,
                    requests_per_day=50000,
                    concurrent_requests=50,
                    retry_after=60,
                    cost_per_request=0.01
                ),
                "gpt-4": RateLimit(
                    requests_per_minute=30,
                    requests_per_hour=1500,
                    requests_per_day=25000,
                    concurrent_requests=25,
                    retry_after=60,
                    cost_per_request=0.02
                )
            },
            "huggingface": {
                "default": RateLimit(
                    requests_per_minute=120,
                    requests_per_hour=6000,
                    requests_per_day=100000,
                    concurrent_requests=100,
                    retry_after=30,
                    cost_per_request=0.005
                )
            }
        }
        
        # Request counters for each provider/model
        self.request_counters: Dict[str, Dict[str, RequestCounter]] = defaultdict(dict)
        
        # Request queues for each provider
        self.request_queues: Dict[str, queue.PriorityQueue] = defaultdict(queue.PriorityQueue)
        
        # Locks for thread safety
        self.counter_locks: Dict[str, threading.Lock] = defaultdict(threading.Lock)
        self.queue_locks: Dict[str, threading.Lock] = defaultdict(threading.Lock)
        
        # Load saved state
        self.load_state()
        
        # Start queue processing threads
        self.queue_processors = {}
        self.start_queue_processors()

    def load_state(self) -> None:
        """Load saved rate limit state."""
        state_file = self.data_dir / "rate_limiter_state.json"
        try:
            if state_file.exists():
                with open(state_file, 'r') as f:
                    data = json.load(f)
                    
                    for provider, models in data.items():
                        for model, counter in models.items():
                            self.request_counters[provider][model] = RequestCounter(
                                minute_count=counter["minute_count"],
                                hour_count=counter["hour_count"],
                                day_count=counter["day_count"],
                                minute_start=datetime.fromisoformat(counter["minute_start"]),
                                hour_start=datetime.fromisoformat(counter["hour_start"]),
                                day_start=datetime.fromisoformat(counter["day_start"]),
                                concurrent=counter["concurrent"]
                            )
        except Exception as e:
            from ..system_monitor import logger
            logger.error(f"Error loading rate limiter state: {e}")

    def save_state(self) -> None:
        """Save current rate limit state."""
        state_file = self.data_dir / "rate_limiter_state.json"
        try:
            data = {}
            for provider, models in self.request_counters.items():
                data[provider] = {}
                for model, counter in models.items():
                    data[provider][model] = {
                        "minute_count": counter.minute_count,
                        "hour_count": counter.hour_count,
                        "day_count": counter.day_count,
                        "minute_start": counter.minute_start.isoformat(),
                        "hour_start": counter.hour_start.isoformat(),
                        "day_start": counter.day_start.isoformat(),
                        "concurrent": counter.concurrent
                    }
            
            temp_file = state_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(data, f)
            temp_file.replace(state_file)
        
        except Exception as e:
            from ..system_monitor import logger
            logger.error(f"Error saving rate limiter state: {e}")
            if temp_file.exists():
                temp_file.unlink()

    def start_queue_processors(self) -> None:
        """Start queue processing threads for each provider."""
        self._shutdown_event = threading.Event()
        for provider in self.rate_limits.keys():
            thread = threading.Thread(
                target=self._process_queue,
                args=(provider,),
                daemon=True
            )
            thread.start()
            self.queue_processors[provider] = thread

    def stop_queue_processors(self) -> None:
        """Stop all queue processing threads."""
        self._shutdown_event.set()
        for thread in self.queue_processors.values():
            thread.join(timeout=5)

    def check_rate_limit(self, provider: str,
                        model: str = "default") -> Tuple[bool, Optional[int]]:
        """
        Check if a request would exceed rate limits.
        
        Args:
            provider (str): Provider name
            model (str): Model name
            
        Returns:
            Tuple[bool, Optional[int]]: (allowed, retry_after_seconds)
        """
        try:
            limits = self.rate_limits[provider][model]
            
            with self.counter_locks[provider]:
                counter = self._get_or_create_counter(provider, model)
                current_time = datetime.now()
                
                # Reset counters if time windows have passed
                if (current_time - counter.minute_start).total_seconds() >= 60:
                    counter.minute_count = 0
                    counter.minute_start = current_time
                
                if (current_time - counter.hour_start).total_seconds() >= 3600:
                    counter.hour_count = 0
                    counter.hour_start = current_time
                
                if (current_time - counter.day_start).total_seconds() >= 86400:
                    counter.day_count = 0
                    counter.day_start = current_time
                
                # Check limits
                if (counter.minute_count >= limits.requests_per_minute or
                    counter.hour_count >= limits.requests_per_hour or
                    counter.day_count >= limits.requests_per_day or
                    counter.concurrent >= limits.concurrent_requests):
                    return False, limits.retry_after
                
                return True, None
        
        except Exception as e:
            from ..system_monitor import logger
            logger.error(f"Error checking rate limit: {e}")
            return False, 60

    def increment_counters(self, provider: str,
                         model: str = "default") -> None:
        """
        Increment request counters.
        
        Args:
            provider (str): Provider name
            model (str): Model name
        """
        try:
            with self.counter_locks[provider]:
                counter = self._get_or_create_counter(provider, model)
                counter.minute_count += 1
                counter.hour_count += 1
                counter.day_count += 1
                counter.concurrent += 1
                # Throttle state saving to avoid excessive disk I/O
                if datetime.now() - self._last_save_time >= self._save_interval:
                    self.save_state()
                    self._last_save_time = datetime.now()
        
        except Exception as e:
            from ..system_monitor import logger
            logger.error(f"Error incrementing counters: {e}")

    def decrement_concurrent(self, provider: str,
                           model: str = "default") -> None:
        """
        Decrement concurrent request counter.
        
        Args:
            provider (str): Provider name
            model (str): Model name
        """
        try:
            with self.counter_locks[provider]:
                counter = self._get_or_create_counter(provider, model)
                counter.concurrent = max(0, counter.concurrent - 1)
                self.save_state()
        
        except Exception as e:
            from ..system_monitor import logger
            logger.error(f"Error decrementing concurrent counter: {e}")

    def queue_request(self, provider: str,
                     model: str,
                     priority: int,
                     request_func,
                     *args,
                     **kwargs) -> None:
        """
        Queue a request for processing.
        
        Args:
            provider (str): Provider name
            model (str): Model name
            priority (int): Request priority (lower = higher priority)
            request_func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
        """
        try:
            with self.queue_locks[provider]:
                self.request_queues[provider].put(
                    (priority, model, request_func, args, kwargs)
                )
        
        except Exception as e:
            from ..system_monitor import logger
            logger.error(f"Error queuing request: {e}")

    def _process_queue(self, provider: str) -> None:
        """
        Process queued requests for a provider.
        
        Args:
            provider (str): Provider name
        """
        while not self._shutdown_event.is_set():
            try:
                # Get next request from queue
                priority, model, request_func, args, kwargs = (
                    self.request_queues[provider].get()
                )
                
                # Check rate limits
                allowed, retry_after = self.check_rate_limit(provider, model)
                
                if allowed:
                    # Execute request
                    self.increment_counters(provider, model)
                    try:
                        request_func(*args, **kwargs)
                    finally:
                        self.decrement_concurrent(provider, model)
                else:
                    # Re-queue request with same priority
                    time.sleep(retry_after)
                    self.queue_request(
                        provider,
                        model,
                        priority,
                        request_func,
                        *args,
                        **kwargs
                    )
            
            except Exception as e:
                from ..system_monitor import logger
                logger.error(f"Error processing queue for {provider}: {e}")
                time.sleep(1)

    def _get_or_create_counter(self, provider: str,
                              model: str) -> RequestCounter:
        """
        Get or create request counter for provider/model.
        
        Args:
            provider (str): Provider name
            model (str): Model name
            
        Returns:
            RequestCounter: Request counter
        """
        if model not in self.request_counters[provider]:
            self.request_counters[provider][model] = RequestCounter(
                minute_count=0,
                hour_count=0,
                day_count=0,
                minute_start=datetime.now(),
                hour_start=datetime.now(),
                day_start=datetime.now(),
                concurrent=0
            )
        return self.request_counters[provider][model]

    def get_usage_stats(self, provider: str = None) -> Dict:
        """
        Get current usage statistics.
        
        Args:
            provider (str, optional): Provider to get stats for
            
        Returns:
            Dict: Usage statistics
        """
        try:
            stats = {}
            
            providers = [provider] if provider else self.rate_limits.keys()
            
            for prov in providers:
                stats[prov] = {}
                for model in self.request_counters[prov]:
                    counter = self.request_counters[prov][model]
                    limits = self.rate_limits[prov].get(
                        model,
                        self.rate_limits[prov]["default"]
                    )
                    
                    stats[prov][model] = {
                        "minute": {
                            "used": counter.minute_count,
                            "limit": limits.requests_per_minute,
                            "remaining": limits.requests_per_minute - counter.minute_count
                        },
                        "hour": {
                            "used": counter.hour_count,
                            "limit": limits.requests_per_hour,
                            "remaining": limits.requests_per_hour - counter.hour_count
                        },
                        "day": {
                            "used": counter.day_count,
                            "limit": limits.requests_per_day,
                            "remaining": limits.requests_per_day - counter.day_count
                        },
                        "concurrent": {
                            "used": counter.concurrent,
                            "limit": limits.concurrent_requests,
                            "remaining": limits.concurrent_requests - counter.concurrent
                        }
                    }
            
            return stats
        
        except Exception as e:
            from ..system_monitor import logger
            logger.error(f"Error getting usage stats: {e}")
            return {}

    def estimate_cost(self, provider: str,
                     model: str = "default",
                     num_requests: int = 1) -> float:
        """
        Estimate cost for number of requests.
        
        Args:
            provider (str): Provider name
            model (str): Model name
            num_requests (int): Number of requests
            
        Returns:
            float: Estimated cost
        """
        try:
            limits = self.rate_limits[provider].get(
                model,
                self.rate_limits[provider]["default"]
            )
            return limits.cost_per_request * num_requests
        
        except Exception as e:
            from ..system_monitor import logger
            logger.error(f"Error estimating cost: {e}")
            return 0.0

    def get_queue_length(self, provider: str) -> int:
        """
        Get current queue length for a provider.
        
        Args:
            provider (str): Provider name
            
        Returns:
            int: Queue length
        """
        try:
            return self.request_queues[provider].qsize()
        except Exception as e:
            from ..system_monitor import logger
            logger.error(f"Error getting queue length: {e}")
            return 0
