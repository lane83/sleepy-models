#!/usr/bin/env python3
"""
Main entry point for Sleepy-Models.
Initializes and coordinates all system components.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, Optional
import json
from datetime import datetime
import signal
import threading
from dotenv import load_dotenv

from usage_tracker import UsageTracker
from model_performance import ModelPerformanceTracker
from config_manager import ConfigurationManager
from interface_components import GradioInterface
from dream_state import DreamState
from memory_manager import MemoryManager
from knowledge_graph import KnowledgeGraph
from graph_operations import GraphOperations
from rate_limiter import RateLimiter
from request_scheduler import RequestScheduler
from model_manager import ModelManager
from system_monitor import SystemMonitor

from provider_adapters.anthropic_adapter import AnthropicAdapter
from provider_adapters.openai_adapter import OpenAIAdapter
from provider_adapters.huggingface_adapter import HuggingFaceAdapter


class SleepyModels:
    """Main application class coordinating all components."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the application.
        
        Args:
            config_path (Optional[str]): Path to configuration file
        """
        # Load environment variables
        load_dotenv()
        
        # Set up logging
        self.setup_logging()
        
        # Initialize data directory
        self.data_dir = Path("data")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Load configuration
        self.config = self.load_config(config_path)
        
        # Initialize components
        self.initialize_components()
        
        # Set up signal handlers
        self.setup_signal_handlers()
        
        # Start monitoring
        self.start_monitoring()
        
        logging.info("SleepyModels initialization complete")

    def setup_logging(self) -> None:
        """Set up logging configuration."""
        log_dir = Path("logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(
                    log_dir / f"sleepy_models_{datetime.now():%Y%m%d}.log"
                ),
                logging.StreamHandler(sys.stdout)
            ]
        )

    def load_config(self, config_path: Optional[str]) -> Dict:
        """
        Load configuration from file or use defaults.
        
        Args:
            config_path (Optional[str]): Path to configuration file
            
        Returns:
            Dict: Configuration dictionary
        """
        default_config = {
            "interface": {
                "host": "0.0.0.0",
                "port": 7860,
                "share": False
            },
            "monitoring": {
                "interval": 60,
                "retention_days": 7
            },
            "memory": {
                "short_term_capacity": 1000,
                "long_term_capacity": 100000
            },
            "dream_state": {
                "min_duration": 60,
                "max_duration": 300
            }
        }
        
        if config_path:
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                    # Validate and merge with defaults
                    for section in user_config:
                        if section in default_config:
                            # Validate each config value
                            for key, value in user_config[section].items():
                                if key in default_config[section]:
                                    # Type checking
                                    expected_type = type(default_config[section][key])
                                    if not isinstance(value, expected_type):
                                        logging.warning(
                                            f"Invalid type for {section}.{key}. "
                                            f"Expected {expected_type}, got {type(value)}"
                                        )
                                        continue
                                    # Range checking for numeric values
                                    if isinstance(value, (int, float)):
                                        if key == "port" and not (1024 <= value <= 65535):
                                            logging.warning(
                                                f"Invalid port number {value}. "
                                                "Using default 7860"
                                            )
                                            continue
                                    default_config[section][key] = value
            except Exception as e:
                logging.error(f"Error loading config file: {e}")
        
        return default_config

    def initialize_components(self) -> None:
        """Initialize all system components."""
        try:
            # Initialize component tracking
            self.components = {}
            
            # Initialize model adapters
            self.adapters = {
                "anthropic": AnthropicAdapter(),
                "openai": OpenAIAdapter(),
                "huggingface": HuggingFaceAdapter()
            }
            
            # Initialize core components
            self.config_manager = ConfigurationManager(
                data_dir=str(self.data_dir)
            )
            self.components["config_manager"] = self.config_manager
            
            self.rate_limiter = RateLimiter(
                data_dir=str(self.data_dir)
            )
            self.components["rate_limiter"] = self.rate_limiter
            
            self.request_scheduler = RequestScheduler(
                self.rate_limiter,
                data_dir=str(self.data_dir)
            )
            self.components["request_scheduler"] = self.request_scheduler
            
            self.usage_tracker = UsageTracker(
                data_dir=str(self.data_dir)
            )
            self.components["usage_tracker"] = self.usage_tracker
            
            self.performance_tracker = ModelPerformanceTracker(
                data_dir=str(self.data_dir)
            )
            self.components["performance_tracker"] = self.performance_tracker
            
            # Initialize knowledge components
            self.knowledge_graph = KnowledgeGraph(
                data_dir=str(self.data_dir)
            )
            self.components["knowledge_graph"] = self.knowledge_graph
            
            self.graph_operations = GraphOperations(
                self.knowledge_graph
            )
            self.components["graph_operations"] = self.graph_operations
            
            self.memory_manager = MemoryManager(
                data_dir=str(self.data_dir)
            )
            self.components["memory_manager"] = self.memory_manager
            
            # Initialize dream state
            self.dream_state = DreamState(
                self.knowledge_graph,
                self.memory_manager,
                data_dir=str(self.data_dir)
            )
            self.components["dream_state"] = self.dream_state
            
            # Initialize model management
            self.model_manager = ModelManager(
                self.rate_limiter,
                self.request_scheduler,
                self.dream_state,
                data_dir=str(self.data_dir)
            )
            self.components["model_manager"] = self.model_manager
            
            # Initialize monitoring
            self.system_monitor = SystemMonitor(
                self.components,
                data_dir=str(self.data_dir)
            )
            
            # Initialize interface
            self.interface = GradioInterface(
                self.usage_tracker,
                self.performance_tracker,
                self.config_manager,
                self.model_manager
            )
            
            logging.info("All components initialized successfully")
        
        except Exception as e:
            logging.error(f"Error initializing components: {e}")
            raise

    def setup_signal_handlers(self) -> None:
        """Set up system signal handlers."""
        def signal_handler(signum, frame):
            logging.info(f"Received signal {signum}")
            self.shutdown()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def start_monitoring(self) -> None:
        """Start system monitoring."""
        try:
            self.system_monitor.start_monitoring(
                interval=self.config["monitoring"]["interval"]
            )
            logging.info("System monitoring started")
        
        except Exception as e:
            logging.error(f"Error starting monitoring: {e}")

    def shutdown(self) -> None:
        """Clean shutdown of all components."""
        try:
            logging.info("Initiating shutdown sequence")
            
            # Stop monitoring
            self.system_monitor.stop_monitoring()
            
            # Save component states with timeout
            def save_component_state(name: str, component: Any) -> None:
                try:
                    if hasattr(component, 'save_state'):
                        component.save_state()
                    logging.info(f"Saved state for {name}")
                except Exception as e:
                    logging.error(f"Error saving state for {name}: {e}")
            
            threads = []
            for name, component in self.components.items():
                t = threading.Thread(
                    target=save_component_state,
                    args=(name, component)
                )
                t.start()
                threads.append(t)
            
            # Wait for all threads with timeout
            for t in threads:
                t.join(timeout=5)  # 5 second timeout per component
                if t.is_alive():
                    logging.warning(f"Timeout saving state for component")
            
            logging.info("Shutdown complete")
        
        except Exception as e:
            logging.error(f"Error during shutdown: {e}")

    def start(self) -> None:
        """Start the application."""
        max_retries = 3
        retry_delay = 5  # seconds
        
        for attempt in range(max_retries):
            try:
                interface = self.interface.create_interface()
                interface.launch(
                    server_name=self.config["interface"]["host"],
                    server_port=self.config["interface"]["port"],
                    share=self.config["interface"]["share"]
                )
                break
            except Exception as e:
                logging.error(
                    f"Error starting application (attempt {attempt + 1}/{max_retries}): {e}"
                )
                if attempt < max_retries - 1:
                    logging.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    logging.error("Max retries reached. Shutting down.")
                    self.shutdown()
                    raise


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Sleepy-Models")
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file"
    )
    args = parser.parse_args()
    
    try:
        app = SleepyModels(config_path=args.config)
        app.start()
    
    except Exception as e:
        logging.error(f"Application error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
