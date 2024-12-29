#!/usr/bin/env python3
"""
Request scheduling system for Sleepy-Models.
Manages request scheduling, load balancing, and fallback handling.
"""

import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Callable, Any
import threading
import queue
from dataclasses import dataclass
import json
from pathlib import Path
import heapq
from collections import defaultdict
import random


@dataclass
class RequestMetrics:
    """Metrics for a model's request performance."""
    success_count: int
    error_count: int
    average_latency: float
    error_rate: float
    last_success: Optional[datetime]
    last_error: Optional[datetime]
    consecutive_errors: int
    token_count: int = 0
    context_window_size: int = 0
    uncertainty_level: float = 0.0
    knowledge_access_count: int = 0


@dataclass
class Request:
    """Represents a scheduled request."""
    id: str
    provider: str
    model: str
    priority: int
    function: Callable
    args: tuple
    kwargs: dict
    timeout: float
    retries: int
    max_retries: int
    fallback_models: List[str]
    callback: Optional[Callable]
    error_callback: Optional[Callable]
    created_at: datetime
    scheduled_for: datetime


class RequestScheduler:
    """Manages request scheduling and execution."""
    
    def __init__(self, rate_limiter, system_monitor, dream_state, data_dir: str = "data"):
        """
        Initialize the request scheduler.
        
        Args:
            rate_limiter: RateLimiter instance
            system_monitor: SystemMonitor instance
            dream_state: DreamState instance
            data_dir (str): Directory to store scheduler data
        """
        self.rate_limiter = rate_limiter
        self.system_monitor = system_monitor
        self.dream_state = dream_state
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Local processing constraints
        self.local_processing_rate = 10  # bits per second
        self.current_bit_rate = 0
        self.bit_rate_lock = threading.Lock()
        self.bit_rate_history = []
        self.bit_rate_window = 60  # seconds
        
        # Request queues by priority
        self.priority_queues: Dict[int, List[Request]] = defaultdict(list)
        
        # Performance metrics for each model
        self.model_metrics: Dict[str, Dict[str, RequestMetrics]] = defaultdict(dict)
        
        # Active requests
        self.active_requests: Dict[str, Request] = {}
        
        # Configuration
        self.config = {
            "max_retries": 3,
            "min_retry_delay": 1.0,  # seconds
            "max_retry_delay": 30.0,  # seconds
            "error_threshold": 0.1,  # 10% error rate
            "consecutive_errors_threshold": 5,
            "health_check_interval": 60,  # seconds
            "load_balance_threshold": 0.8,  # 80% capacity
            "request_timeout": 30.0  # seconds
        }
        
        # Locks
        self.queue_lock = threading.Lock()
        self.metrics_lock = threading.Lock()
        
        # Background threads
        self.scheduler_thread = threading.Thread(
            target=self._schedule_requests,
            daemon=True
        )
        self.health_check_thread = threading.Thread(
            target=self._check_model_health,
            daemon=True
        )
        
        # Load saved metrics
        self.load_metrics()
        
        # Start background threads
        self.scheduler_thread.start()
        self.health_check_thread.start()

    def load_metrics(self) -> None:
        """Load saved model metrics."""
        metrics_file = self.data_dir / "model_metrics.json"
        try:
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    data = json.load(f)
                    
                    for provider, models in data.items():
                        for model, metrics in models.items():
                            self.model_metrics[provider][model] = RequestMetrics(
                                success_count=metrics["success_count"],
                                error_count=metrics["error_count"],
                                average_latency=metrics["average_latency"],
                                error_rate=metrics["error_rate"],
                                last_success=datetime.fromisoformat(metrics["last_success"])
                                    if metrics["last_success"] else None,
                                last_error=datetime.fromisoformat(metrics["last_error"])
                                    if metrics["last_error"] else None,
                                consecutive_errors=metrics["consecutive_errors"]
                            )
        except Exception as e:
            print(f"Error loading model metrics: {e}")

    def save_metrics(self) -> None:
        """Save current model metrics."""
        metrics_file = self.data_dir / "model_metrics.json"
        try:
            data = {}
            for provider, models in self.model_metrics.items():
                data[provider] = {}
                for model, metrics in models.items():
                    data[provider][model] = {
                        "success_count": metrics.success_count,
                        "error_count": metrics.error_count,
                        "average_latency": metrics.average_latency,
                        "error_rate": metrics.error_rate,
                        "last_success": metrics.last_success.isoformat()
                            if metrics.last_success else None,
                        "last_error": metrics.last_error.isoformat()
                            if metrics.last_error else None,
                        "consecutive_errors": metrics.consecutive_errors,
                        "token_count": metrics.token_count,
                        "context_window_size": metrics.context_window_size,
                        "uncertainty_level": metrics.uncertainty_level,
                        "knowledge_access_count": metrics.knowledge_access_count
                    }
            
            temp_file = metrics_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(data, f)
            temp_file.replace(metrics_file)
        
        except Exception as e:
            print(f"Error saving model metrics: {e}")
            if temp_file.exists():
                temp_file.unlink()

    def schedule_request(self,
                        provider: str,
                        model: str,
                        function: Callable,
                        priority: int = 1,
                        timeout: float = None,
                        max_retries: int = None,
                        fallback_models: List[str] = None,
                        callback: Callable = None,
                        error_callback: Callable = None,
                        *args,
                        **kwargs) -> str:
        """
        Schedule a request for execution.
        
        Args:
            provider (str): Provider name
            model (str): Model name
            function (Callable): Function to execute
            priority (int): Request priority (lower = higher priority)
            timeout (float, optional): Request timeout
            max_retries (int, optional): Maximum retry attempts
            fallback_models (List[str], optional): Fallback models
            callback (Callable, optional): Success callback
            error_callback (Callable, optional): Error callback
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            str: Request ID
        """
        try:
            request_id = f"req_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            
            request = Request(
                id=request_id,
                provider=provider,
                model=model,
                priority=priority,
                function=function,
                args=args,
                kwargs=kwargs,
                timeout=timeout or self.config["request_timeout"],
                retries=0,
                max_retries=max_retries or self.config["max_retries"],
                fallback_models=fallback_models or [],
                callback=callback,
                error_callback=error_callback,
                created_at=datetime.now(),
                scheduled_for=datetime.now()
            )
            
            with self.queue_lock:
                heapq.heappush(self.priority_queues[priority], (request.scheduled_for, request))
            
            return request_id
        
        except Exception as e:
            print(f"Error scheduling request: {e}")
            return None

    def cancel_request(self, request_id: str) -> bool:
        """
        Cancel a scheduled request.
        
        Args:
            request_id (str): Request ID
            
        Returns:
            bool: Success status
        """
        try:
            with self.queue_lock:
                # Check active requests
                if request_id in self.active_requests:
                    del self.active_requests[request_id]
                    return True
                
                # Check queued requests
                for priority in self.priority_queues:
                    for i, (_, request) in enumerate(self.priority_queues[priority]):
                        if request.id == request_id:
                            self.priority_queues[priority].pop(i)
                            heapq.heapify(self.priority_queues[priority])
                            return True
                
                return False
        
        except Exception as e:
            print(f"Error canceling request: {e}")
            return False

    def get_request_status(self, request_id: str) -> Optional[Dict]:
        """
        Get status of a request.
        
        Args:
            request_id (str): Request ID
            
        Returns:
            Optional[Dict]: Request status information
        """
        try:
            # Check active requests
            if request_id in self.active_requests:
                request = self.active_requests[request_id]
                return {
                    "status": "active",
                    "provider": request.provider,
                    "model": request.model,
                    "priority": request.priority,
                    "retries": request.retries,
                    "created_at": request.created_at.isoformat(),
                    "scheduled_for": request.scheduled_for.isoformat()
                }
            
            # Check queued requests
            for priority in self.priority_queues:
                for scheduled_time, request in self.priority_queues[priority]:
                    if request.id == request_id:
                        return {
                            "status": "queued",
                            "provider": request.provider,
                            "model": request.model,
                            "priority": request.priority,
                            "retries": request.retries,
                            "created_at": request.created_at.isoformat(),
                            "scheduled_for": scheduled_time.isoformat()
                        }
            
            return None
        
        except Exception as e:
            print(f"Error getting request status: {e}")
            return None

    def update_model_metrics(self,
                           provider: str,
                           model: str,
                           success: bool,
                           latency: float,
                           token_count: int = 0,
                           context_window_size: int = 0,
                           uncertainty_level: float = 0.0,
                           knowledge_access_count: int = 0) -> None:
        """
        Update performance metrics for a model.
        
        Args:
            provider (str): Provider name
            model (str): Model name
            success (bool): Whether request was successful
            latency (float): Request latency
            token_count (int): Number of tokens processed
            context_window_size (int): Size of context window used
            uncertainty_level (float): Model's uncertainty level (0.0-1.0)
            knowledge_access_count (int): Number of knowledge graph accesses
        """
        try:
            with self.metrics_lock:
                if model not in self.model_metrics[provider]:
                    self.model_metrics[provider][model] = RequestMetrics(
                        success_count=0,
                        error_count=0,
                        average_latency=0.0,
                        error_rate=0.0,
                        last_success=None,
                        last_error=None,
                        consecutive_errors=0,
                        token_count=0,
                        context_window_size=0,
                        uncertainty_level=0.0,
                        knowledge_access_count=0
                    )
                
                metrics = self.model_metrics[provider][model]
                
                if success:
                    metrics.success_count += 1
                    metrics.last_success = datetime.now()
                    metrics.consecutive_errors = 0
                    
                    # Update success-related metrics
                    metrics.token_count += token_count
                    metrics.context_window_size = max(
                        metrics.context_window_size,
                        context_window_size
                    )
                    metrics.uncertainty_level = (
                        (metrics.uncertainty_level * (metrics.success_count - 1) + uncertainty_level) /
                        metrics.success_count
                    )
                    metrics.knowledge_access_count += knowledge_access_count
                else:
                    metrics.error_count += 1
                    metrics.last_error = datetime.now()
                    metrics.consecutive_errors += 1
                
                total_requests = metrics.success_count + metrics.error_count
                metrics.error_rate = metrics.error_count / total_requests
                
                # Update average latency
                metrics.average_latency = (
                    (metrics.average_latency * (total_requests - 1) + latency) /
                    total_requests
                )
                
                self.save_metrics()
        
        except Exception as e:
            print(f"Error updating model metrics: {e}")

    def _check_model_health(self) -> None:
        """Monitor model health and adjust scheduling."""
        while True:
            try:
                with self.metrics_lock:
                    for provider in self.model_metrics:
                        for model, metrics in self.model_metrics[provider].items():
                            # Check error rate
                            if metrics.error_rate > self.config["error_threshold"]:
                                print(f"High error rate for {provider}/{model}: {metrics.error_rate:.2%}")
                            
                            # Check consecutive errors
                            if metrics.consecutive_errors >= self.config["consecutive_errors_threshold"]:
                                print(f"Consecutive errors for {provider}/{model}: {metrics.consecutive_errors}")
                            
                            # Check if model needs cool-down
                            if self._needs_cooldown(metrics):
                                self._apply_cooldown(provider, model)
                
                time.sleep(self.config["health_check_interval"])
            
            except Exception as e:
                print(f"Error in health check: {e}")
                time.sleep(1)

    def _needs_cooldown(self, metrics: RequestMetrics) -> bool:
        """
        Check if a model needs cool-down period.
        
        Args:
            metrics (RequestMetrics): Model metrics
            
        Returns:
            bool: Whether cool-down is needed
        """
        return (metrics.consecutive_errors >= self.config["consecutive_errors_threshold"] or
                metrics.error_rate > self.config["error_threshold"])

    def _apply_cooldown(self, provider: str, model: str) -> None:
        """
        Apply cool-down period to a model.
        
        Args:
            provider (str): Provider name
            model (str): Model name
        """
        try:
            # Reschedule active requests using this model
            with self.queue_lock:
                for request_id, request in list(self.active_requests.items()):
                    if request.provider == provider and request.model == model:
                        if request.fallback_models:
                            # Try fallback model
                            fallback = request.fallback_models.pop(0)
                            request.model = fallback
                            request.retries = 0
                        else:
                            # No fallback available, retry later
                            request.scheduled_for = datetime.now() + timedelta(
                                seconds=self.config["max_retry_delay"]
                            )
                            heapq.heappush(
                                self.priority_queues[request.priority],
                                (request.scheduled_for, request)
                            )
                        del self.active_requests[request_id]
        
        except Exception as e:
            print(f"Error applying cooldown: {e}")

    def _schedule_requests(self) -> None:
        """Process and schedule queued requests."""
        while True:
            try:
                with self.queue_lock:
                    now = datetime.now()
                    
                    # Process each priority level
                    for priority in sorted(self.priority_queues.keys()):
                        queue = self.priority_queues[priority]
                        
                        while queue and queue[0][0] <= now:
                            _, request = heapq.heappop(queue)
                            
                            # Check rate limits
                            allowed, retry_after = self.rate_limiter.check_rate_limit(
                                request.provider,
                                request.model
                            )
                            
                            if allowed:
                                # Execute request
                                self._execute_request(request)
                            else:
                                # Reschedule with delay
                                request.scheduled_for = now + timedelta(seconds=retry_after)
                                heapq.heappush(queue, (request.scheduled_for, request))
                
                time.sleep(0.1)  # Prevent tight loop
            
            except Exception as e:
                print(f"Error in request scheduler: {e}")
                time.sleep(1)

    def _execute_request(self, request: Request) -> None:
        """
        Execute a scheduled request.
        
        Args:
            request (Request): Request to execute
        """
        try:
            self.active_requests[request.id] = request
            
            # Start execution in separate thread
            thread = threading.Thread(
                target=self._handle_request_execution,
                args=(request,),
                daemon=True
            )
            thread.start()
        
        except Exception as e:
            print(f"Error executing request: {e}")
            self._handle_request_error(request, e)

    def _handle_request_execution(self, request: Request) -> None:
        """
        Handle the execution of a request.
        
        Args:
            request (Request): Request to execute
        """
        try:
            start_time = time.time()
            
            # Execute request with timeout
            result = self._execute_with_timeout(request)
            
            # Calculate latency
            latency = time.time() - start_time
            
            # Collect additional metrics from the result
            token_count = getattr(result, 'token_count', 0)
            context_window_size = getattr(result, 'context_window_size', 0)
            uncertainty_level = getattr(result, 'uncertainty_level', 0.0)
            knowledge_access_count = getattr(result, 'knowledge_access_count', 0)
            
            # Update metrics with all collected data
            self.update_model_metrics(
                request.provider,
                request.model,
                True,
                latency,
                token_count=token_count,
                context_window_size=context_window_size,
                uncertainty_level=uncertainty_level,
                knowledge_access_count=knowledge_access_count
            )
            
            # Estimate request size in bits
            request_size = sum(len(str(arg)) for arg in request.args) * 8
            request_size += sum(len(str(val)) for val in request.kwargs.values()) * 8
            
            # Update bit rate
            self._update_bit_rate(request_size)
            
            # Call success callback
            if request.callback:
                request.callback(result)
            
            # Cleanup
            if request.id in self.active_requests:
                del self.active_requests[request.id]
        
        except Exception as e:
            self._handle_request_error(request, e)

    def _execute_with_timeout(self, request: Request) -> Any:
        """
        Execute a request with timeout.
        
        Args:
            request (Request): Request to execute
            
        Returns:
            Any: Request result
            
        Raises:
            TimeoutError: If request exceeds timeout
        """
        result_queue = queue.Queue()
        
        def target():
            try:
                result = request.function(*request.args, **request.kwargs)
                result_queue.put(("success", result))
            except Exception as e:
                result_queue.put(("error", e))
        
        thread = threading.Thread(target=target)
        thread.daemon = True
        thread.start()
        
        try:
            status, result = result_queue.get(timeout=request.timeout)
            if status == "error":
                raise result
            return result
        except queue.Empty:
            raise TimeoutError(f"Request timed out after {request.timeout} seconds")

    def _handle_request_error(self, request: Request, error: Exception) -> None:
        """
        Handle request execution error.
        
        Args:
            request (Request): Failed request
            error (Exception): Error that occurred
        """
        try:
            # Update metrics
            self.update_model_metrics(
                request.provider,
                request.model,
                False,
                0.0
            )
            
            # Remove from active requests
            if request.id in self.active_requests:
                del self.active_requests[request.id]
            
            # Check if we should retry
            if request.retries < request.max_retries:
                self._schedule_retry(request)
            
            # Try fallback model
            elif request.fallback_models:
                self._try_fallback(request)
            
            # Handle final failure
            else:
                if request.error_callback:
                    request.error_callback(error)
        
        except Exception as e:
            print(f"Error handling request failure: {e}")

    def _schedule_retry(self, request: Request) -> None:
        """
        Schedule a request retry.
        
        Args:
            request (Request): Request to retry
        """
        try:
            # Calculate retry delay
            delay = min(
                self.config["max_retry_delay"],
                self.config["min_retry_delay"] * (2 ** request.retries)
            )
            
            # Update request for retry
            request.retries += 1
            request.scheduled_for = datetime.now() + timedelta(seconds=delay)
            
            # Re-queue request
            with self.queue_lock:
                heapq.heappush(
                    self.priority_queues[request.priority],
                    (request.scheduled_for, request)
                )
        
        except Exception as e:
            print(f"Error scheduling retry: {e}")

    def _try_fallback(self, request: Request) -> None:
        """
        Try executing request with fallback model.
        
        Args:
            request (Request): Failed request
        """
        try:
            # Get next fallback model
            fallback_model = request.fallback_models.pop(0)
            
            # Reset retry count for new model
            request.model = fallback_model
            request.retries = 0
            request.scheduled_for = datetime.now()
            
            # Schedule with fallback model
            with self.queue_lock:
                heapq.heappush(
                    self.priority_queues[request.priority],
                    (request.scheduled_for, request)
                )
        
        except Exception as e:
            print(f"Error trying fallback model: {e}")

    def _update_bit_rate(self, bits: int) -> None:
        """
        Update the current bit rate and maintain history.
        
        Args:
            bits (int): Number of bits processed
        """
        with self.bit_rate_lock:
            now = time.time()
            # Remove old entries from history
            self.bit_rate_history = [
                (timestamp, b) for timestamp, b in self.bit_rate_history
                if now - timestamp <= self.bit_rate_window
            ]
            
            # Add new entry
            self.bit_rate_history.append((now, bits))
            self.current_bit_rate = sum(b for _, b in self.bit_rate_history) / self.bit_rate_window
            
            # Check if we need to trigger a dream state
            if self.current_bit_rate > self.local_processing_rate:
                self._trigger_dream_state()

    def _trigger_dream_state(self) -> None:
        """
        Trigger a dream state to consolidate memory and reduce processing load.
        """
        try:
            # Save current context
            self.dream_state.save_context()
            
            # Reduce processing rate
            with self.bit_rate_lock:
                self.current_bit_rate = 0
                self.bit_rate_history = []
            
            # Initiate dream state
            self.dream_state.start()
            
        except Exception as e:
            print(f"Error triggering dream state: {e}")

    def get_queue_stats(self) -> Dict:
        """
        Get current queue statistics.
        
        Returns:
            Dict: Queue statistics including bit rate
        """
        try:
            stats = {
                "active_requests": len(self.active_requests),
                "queued_requests": sum(
                    len(queue) for queue in self.priority_queues.values()
                ),
                "priority_distribution": {
                    priority: len(queue)
                    for priority, queue in self.priority_queues.items()
                },
                "model_usage": defaultdict(lambda: defaultdict(int))
            }
            
            # Count requests per model
            for request in self.active_requests.values():
                stats["model_usage"][request.provider][request.model] += 1
            
            for queue in self.priority_queues.values():
                for _, request in queue:
                    stats["model_usage"][request.provider][request.model] += 1
            
            return stats
        
        except Exception as e:
            print(f"Error getting queue stats: {e}")
            return {}
