#!/usr/bin/env python3
"""
System monitoring for Sleepy-Models.
Tracks system health, resource usage, and performance metrics.
"""

import os
import psutil
import threading
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np
import torch
import logging
from queue import Queue
import time


@dataclass
class SystemMetrics:
    """System performance metrics."""
    cpu_percent: float
    memory_percent: float
    gpu_utilization: Optional[float]
    gpu_memory_used: Optional[float]
    disk_usage_percent: float
    network_io: Dict[str, int]
    process_count: int
    timestamp: datetime


@dataclass
class ComponentStatus:
    """Component health status."""
    name: str
    status: str  # 'healthy', 'degraded', 'failed'
    last_check: datetime
    metrics: Dict[str, Any]
    errors: List[str]


class SystemMonitor:
    """Monitors system health and performance."""
    
    def __init__(self, 
                 components: Dict,
                 data_dir: str = "data",
                 log_file: str = "system_monitor.log"):
        """
        Initialize system monitor.
        
        Args:
            components: Dict of system components to monitor
            data_dir (str): Directory to store monitoring data
            log_file (str): Log file name
        """
        self.components = components
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up logging
        self.log_file = self.data_dir / log_file
        logging.basicConfig(
            filename=str(self.log_file),
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Metrics storage
        self.metrics_file = self.data_dir / "system_metrics.json"
        self.system_metrics: List[SystemMetrics] = []
        self.component_status: Dict[str, ComponentStatus] = {}
        
        # Load alert thresholds
        self.thresholds = self._load_thresholds()
        
        # Monitoring state
        self.is_monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.alert_queue: Queue = Queue()
        
        # Load previous metrics
        self.load_metrics()
        
        # Initialize GPU monitoring
        self.gpu_available = False
        self.gpu_count = 0
        try:
            import torch
            self.gpu_available = torch.cuda.is_available()
            if self.gpu_available:
                self.gpu_count = torch.cuda.device_count()
        except ImportError:
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                if gpus:
                    self.gpu_available = True
                    self.gpu_count = len(gpus)
            except ImportError:
                pass

    def _load_thresholds(self) -> Dict[str, float]:
        """Load alert thresholds from config file or use defaults."""
        thresholds_file = self.data_dir / "thresholds.json"
        default_thresholds = {
            "cpu_percent": 90.0,
            "memory_percent": 90.0,
            "gpu_utilization": 95.0,
            "gpu_memory_used": 90.0,
            "disk_usage_percent": 90.0,
            "error_rate": 0.1
        }
        
        if thresholds_file.exists():
            try:
                with open(thresholds_file, 'r') as f:
                    return {**default_thresholds, **json.load(f)}
            except Exception as e:
                logging.error(f"Error loading thresholds: {e}")
                return default_thresholds
        return default_thresholds

    def start_monitoring(self, interval: float = 60.0) -> None:
        """
        Start system monitoring.
        
        Args:
            interval (float): Monitoring interval in seconds
        """
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
        logging.info("System monitoring started")

    def stop_monitoring(self) -> None:
        """Stop system monitoring."""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
            self.monitor_thread = None
        logging.info("System monitoring stopped")

    def _monitoring_loop(self, interval: float) -> None:
        """
        Main monitoring loop.
        
        Args:
            interval (float): Monitoring interval in seconds
        """
        while self.is_monitoring:
            try:
                # Collect system metrics
                metrics = self._collect_system_metrics()
                self.system_metrics.append(metrics)
                
                # Check component health
                self._check_component_health()
                
                # Process alerts
                self._process_alerts()
                
                # Save metrics periodically
                self.save_metrics()
                
                # Cleanup old metrics
                self._cleanup_old_metrics()
                
                time.sleep(interval)
            
            except Exception as e:
                logging.error(f"Error in monitoring loop: {e}")
                time.sleep(1)

    def _collect_system_metrics(self) -> SystemMetrics:
        """
        Collect current system metrics.
        
        Returns:
            SystemMetrics: Current system metrics
        """
        try:
            # Get CPU and memory usage
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            # Get GPU metrics if available
            if self.gpu_available:
                gpu_util = torch.cuda.utilization()
                gpu_memory = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
            else:
                gpu_util = None
                gpu_memory = None
            
            # Get disk usage
            disk = psutil.disk_usage('/')
            
            # Get network I/O
            net_io = psutil.net_io_counters()._asdict()
            
            # Get process count
            process_count = len(psutil.pids())
            
            return SystemMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                gpu_utilization=gpu_util,
                gpu_memory_used=gpu_memory,
                disk_usage_percent=disk.percent,
                network_io=net_io,
                process_count=process_count,
                timestamp=datetime.now()
            )
        
        except Exception as e:
            logging.error(f"Error collecting system metrics: {e}")
            raise

    def _check_component_health(self) -> None:
        """Check health of monitored components."""
        try:
            for name, component in self.components.items():
                try:
                    # Get component metrics
                    metrics = component.get_usage_stats()
                    
                    # Check error rate
                    error_rate = (
                        metrics.get("error_count", 0) /
                        max(metrics.get("total_requests", 1), 1)
                    )
                    
                    # Determine status
                    if error_rate > self.thresholds["error_rate"]:
                        status = "degraded"
                        errors = [f"High error rate: {error_rate:.2%}"]
                    else:
                        status = "healthy"
                        errors = []
                    
                    # Update component status
                    self.component_status[name] = ComponentStatus(
                        name=name,
                        status=status,
                        last_check=datetime.now(),
                        metrics=metrics,
                        errors=errors
                    )
                
                except Exception as e:
                    logging.error(f"Error checking {name} health: {e}")
                    self.component_status[name] = ComponentStatus(
                        name=name,
                        status="failed",
                        last_check=datetime.now(),
                        metrics={},
                        errors=[str(e)]
                    )
        
        except Exception as e:
            logging.error(f"Error checking component health: {e}")

    def _process_alerts(self) -> None:
        """Process system alerts."""
        try:
            alerts = []
            
            # Check CPU usage
            if self.system_metrics[-1].cpu_percent > self.thresholds["cpu_percent"]:
                alerts.append(f"High CPU usage: {self.system_metrics[-1].cpu_percent}%")
            
            # Check memory usage
            if self.system_metrics[-1].memory_percent > self.thresholds["memory_percent"]:
                alerts.append(f"High memory usage: {self.system_metrics[-1].memory_percent}%")
            
            # Check GPU usage
            if (self.gpu_available and
                self.system_metrics[-1].gpu_utilization is not None):
                if (self.system_metrics[-1].gpu_utilization >
                    self.thresholds["gpu_utilization"]):
                    alerts.append(
                        f"High GPU utilization: "
                        f"{self.system_metrics[-1].gpu_utilization}%"
                    )
            
            # Check disk usage
            if (self.system_metrics[-1].disk_usage_percent >
                self.thresholds["disk_usage_percent"]):
                alerts.append(
                    f"High disk usage: "
                    f"{self.system_metrics[-1].disk_usage_percent}%"
                )
            
            # Check component health
            for status in self.component_status.values():
                if status.status != "healthy":
                    alerts.append(
                        f"Component {status.name} is {status.status}: "
                        f"{', '.join(status.errors)}"
                    )
            
            # Add alerts to queue
            for alert in alerts:
                self.alert_queue.put(alert)
                logging.warning(alert)
        
        except Exception as e:
            logging.error(f"Error processing alerts: {e}")

    def get_alerts(self) -> List[str]:
        """
        Get pending system alerts.
        
        Returns:
            List[str]: List of alert messages
        """
        alerts = []
        while not self.alert_queue.empty():
            alerts.append(self.alert_queue.get())
        return alerts

    def get_system_status(self) -> Dict[str, Any]:
        """
        Get current system status.
        
        Returns:
            Dict[str, Any]: System status information
        """
        try:
            if not self.system_metrics:
                return {}
            
            current = self.system_metrics[-1]
            
            return {
                "timestamp": current.timestamp.isoformat(),
                "cpu_usage": current.cpu_percent,
                "memory_usage": current.memory_percent,
                "gpu_utilization": current.gpu_utilization,
                "gpu_memory_used": current.gpu_memory_used,
                "disk_usage": current.disk_usage_percent,
                "network_io": current.network_io,
                "process_count": current.process_count,
                "components": {
                    name: {
                        "status": status.status,
                        "last_check": status.last_check.isoformat(),
                        "errors": status.errors
                    }
                    for name, status in self.component_status.items()
                }
            }
        
        except Exception as e:
            logging.error(f"Error getting system status: {e}")
            return {}

    def get_resource_usage(self,
                          time_window: timedelta = timedelta(hours=1)
                          ) -> Dict[str, List[float]]:
        """
        Get resource usage over time window.
        
        Args:
            time_window (timedelta): Time window to analyze
            
        Returns:
            Dict[str, List[float]]: Resource usage metrics
        """
        try:
            cutoff = datetime.now() - time_window
            metrics = [m for m in self.system_metrics if m.timestamp > cutoff]
            
            if not metrics:
                return {}
            
            return {
                "timestamps": [m.timestamp.isoformat() for m in metrics],
                "cpu_usage": [m.cpu_percent for m in metrics],
                "memory_usage": [m.memory_percent for m in metrics],
                "gpu_utilization": [m.gpu_utilization for m in metrics
                                  if m.gpu_utilization is not None],
                "disk_usage": [m.disk_usage_percent for m in metrics]
            }
        
        except Exception as e:
            logging.error(f"Error getting resource usage: {e}")
            return {}

    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get system performance metrics.
        
        Returns:
            Dict[str, Any]: Performance metrics
        """
        try:
            metrics = {}
            
            # Get component metrics
            for name, component in self.components.items():
                try:
                    stats = component.get_usage_stats()
                    metrics[name] = {
                        "requests": stats.get("total_requests", 0),
                        "errors": stats.get("error_count", 0),
                        "average_latency": stats.get("average_latency", 0.0),
                        "total_tokens": stats.get("total_tokens", 0),
                        "cache_hit_rate": (
                            stats.get("cache_hits", 0) /
                            max(stats.get("total_requests", 1), 1)
                        )
                    }
                except Exception as e:
                    logging.error(f"Error getting {name} metrics: {e}")
            
            return metrics
        
        except Exception as e:
            logging.error(f"Error getting performance metrics: {e}")
            return {}

    def save_metrics(self) -> None:
        """Save system metrics to file."""
        try:
            data = {
                "system_metrics": [
                    {
                        "cpu_percent": metrics.cpu_percent,
                        "memory_percent": metrics.memory_percent,
                        "gpu_utilization": metrics.gpu_utilization,
                        "gpu_memory_used": metrics.gpu_memory_used,
                        "disk_usage_percent": metrics.disk_usage_percent,
                        "network_io": metrics.network_io,
                        "process_count": metrics.process_count,
                        "timestamp": metrics.timestamp.isoformat()
                    }
                    for metrics in self.system_metrics
                ],
                "component_status": {
                    name: {
                        "status": status.status,
                        "last_check": status.last_check.isoformat(),
                        "metrics": status.metrics,
                        "errors": status.errors
                    }
                    for name, status in self.component_status.items()
                }
            }
            
            temp_file = self.metrics_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(data, f)
            temp_file.replace(self.metrics_file)
        
        except Exception as e:
            logging.error(f"Error saving metrics: {e}")
            if temp_file.exists():
                temp_file.unlink()

    def load_metrics(self) -> None:
        """Load system metrics from file."""
        try:
            if not self.metrics_file.exists():
                return
            
            with open(self.metrics_file, 'r') as f:
                data = json.load(f)
            
            # Load system metrics
            self.system_metrics = [
                SystemMetrics(
                    cpu_percent=m["cpu_percent"],
                    memory_percent=m["memory_percent"],
                    gpu_utilization=m["gpu_utilization"],
                    gpu_memory_used=m["gpu_memory_used"],
                    disk_usage_percent=m["disk_usage_percent"],
                    network_io=m["network_io"],
                    process_count=m["process_count"],
                    timestamp=datetime.fromisoformat(m["timestamp"])
                )
                for m in data["system_metrics"]
            ]
            
            # Load component status
            self.component_status = {
                name: ComponentStatus(
                    name=name,
                    status=status["status"],
                    last_check=datetime.fromisoformat(status["last_check"]),
                    metrics=status["metrics"],
                    errors=status["errors"]
                )
                for name, status in data["component_status"].items()
            }
        
        except Exception as e:
            logging.error(f"Error loading metrics: {e}")

    def _cleanup_old_metrics(self, max_age: timedelta = timedelta(days=7)) -> None:
        """
        Clean up old metrics data.
        
        Args:
            max_age (timedelta): Maximum age of metrics to keep
        """
        try:
            cutoff = datetime.now() - max_age
            self.system_metrics = [
                m for m in self.system_metrics
                if m.timestamp > cutoff
            ]
            self.save_metrics()
        
        except Exception as e:
            logging.error(f"Error cleaning up metrics: {e}")

    def update_thresholds(self, new_thresholds: Dict[str, float]) -> None:
        """
        Update alert thresholds.
        
        Args:
            new_thresholds (Dict[str, float]): New threshold values
        """
        try:
            for key, value in new_thresholds.items():
                if key in self.thresholds:
                    self.thresholds[key] = value
            logging.info(f"Updated alert thresholds: {self.thresholds}")
        
        except Exception as e:
            logging.error(f"Error updating thresholds: {e}")

    def get_component_metrics(self, component_name: str) -> Optional[Dict[str, Any]]:
        """
        Get metrics for specific component.
        
        Args:
            component_name (str): Component name
            
        Returns:
            Optional[Dict[str, Any]]: Component metrics
        """
        try:
            status = self.component_status.get(component_name)
            if not status:
                return None
            
            return {
                "status": status.status,
                "last_check": status.last_check.isoformat(),
                "metrics": status.metrics,
                "errors": status.errors
            }
        
        except Exception as e:
            logging.error(f"Error getting component metrics: {e}")
            return None

    def get_system_load(self) -> Dict[str, float]:
        """
        Get current system load metrics.
        
        Returns:
            Dict[str, float]: System load metrics
        """
        try:
            # Get system load averages
            load_1, load_5, load_15 = psutil.getloadavg()
            
            # Get CPU count for load per CPU calculation
            cpu_count = psutil.cpu_count()
            
            return {
                "load_1min": load_1,
                "load_5min": load_5,
                "load_15min": load_15,
                "load_per_cpu_1min": load_1 / cpu_count,
                "load_per_cpu_5min": load_5 / cpu_count,
                "load_per_cpu_15min": load_15 / cpu_count
            }
        
        except Exception as e:
            logging.error(f"Error getting system load: {e}")
            return {}
