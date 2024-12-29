import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
from filelock import FileLock


class UsageTracker:
    """Tracks usage metrics and costs for model interactions."""
    
    def __init__(self, data_dir: str = "data", rate_limiter=None, context_manager=None):
        """
        Initialize the usage tracker.
        
        Args:
            data_dir (str): Directory to store usage data
            rate_limiter: Reference to rate limiter instance
            context_manager: Reference to context manager instance
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        self.usage_file = os.path.join(data_dir, "usage_history.json")
        self.lock_file = os.path.join(data_dir, "usage_history.lock")
        self.file_lock = FileLock(self.lock_file)
        
        self.rate_limiter = rate_limiter
        self.context_manager = context_manager
        
        # Cost rates per token for different models
        self.cost_rates = {
            "anthropic/claude-3-sonnet": {
                "input": 0.000015,
                "output": 0.000075
            },
            "anthropic/claude-3-opus": {
                "input": 0.000030,
                "output": 0.000150
            },
            "openai/gpt-4": {
                "input": 0.000030,
                "output": 0.000060
            },
            "microsoft/phi-2": {
                "input": 0.000002,
                "output": 0.000004
            }
        }
        self.load_history()

    def load_history(self) -> None:
        """Load usage history from file with proper error handling."""
        with self.file_lock:
            try:
                if os.path.exists(self.usage_file):
                    with open(self.usage_file, 'r') as f:
                        self.history = json.load(f)
                else:
                    self.history = {
                        "sessions": [],
                        "models": {}
                    }
            except Exception as e:
                print(f"Error loading usage history: {e}")
                self.history = {
                    "sessions": [],
                    "models": {}
                }

    def save_history(self) -> None:
        """Save usage history to file with atomic write operations."""
        with self.file_lock:
            try:
                # Create temp file
                temp_file = f"{self.usage_file}.tmp"
                with open(temp_file, 'w') as f:
                    json.dump(self.history, f)
                # Atomic rename
                os.replace(temp_file, self.usage_file)
            except Exception as e:
                print(f"Error saving usage history: {e}")
                if os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                    except:
                        pass

    def start_session(self, model_name: str, provider: str, task_type: str = "general") -> str:
        """
        Start a new usage tracking session.
        
        Args:
            model_name (str): Name of the model being used
            provider (str): Name of the provider (openai, anthropic, etc.)
            task_type (str): Type of task being performed
            
        Returns:
            str: Session ID for the new session
        """
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        session = {
            "id": session_id,
            "model": model_name,
            "provider": provider,
            "task_type": task_type,
            "start_time": datetime.now().isoformat(),
            "input_tokens": 0,
            "output_tokens": 0,
            "requests": 0,
            "context_size": 0,
            "memory_usage": 0,
            "rate_limit_hits": 0,
            "uncertainty_levels": [],
            "knowledge_access_count": 0,
            "coherence_scores": [],
            "end_time": None,
            "remaining_capacity": None,
            "time_between_requests": [],
            "content_lengths": [],
            "knowledge_access_patterns": [],
            "latency_breakdown": {
                "processing": 0.0,
                "network": 0.0,
                "total": 0.0
            }
        }
        
        if model_name not in self.history["models"]:
            self.history["models"][model_name] = {
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_requests": 0,
                "total_cost": 0.0,
                "average_response_time": 0.0
            }
        
        self.history["sessions"].append(session)
        self.save_history()
        return session_id

    def update_session(self, session_id: str, input_tokens: int, 
                     output_tokens: int, response_time: float,
                     uncertainty: Optional[float] = None,
                     knowledge_access: bool = False,
                     coherence_score: Optional[float] = None) -> None:
        """
        Update session with new usage data.
        
        Args:
            session_id (str): ID of the session to update
            input_tokens (int): Number of input tokens used
            output_tokens (int): Number of output tokens generated
            response_time (float): Time taken for the response
            uncertainty (Optional[float]): Model's uncertainty level
            knowledge_access (bool): Whether knowledge graph was accessed
            coherence_score (Optional[float]): Response coherence score
        """
        try:
            for session in self.history["sessions"]:
                if session["id"] == session_id:
                    session["input_tokens"] += input_tokens
                    session["output_tokens"] += output_tokens
                    session["requests"] += 1
                    
                    model = session["model"]
                    self.history["models"][model]["total_input_tokens"] += input_tokens
                    self.history["models"][model]["total_output_tokens"] += output_tokens
                    self.history["models"][model]["total_requests"] += 1
                    
                    # Update average response time
                    current_avg = self.history["models"][model]["average_response_time"]
                    total_requests = self.history["models"][model]["total_requests"]
                    new_avg = ((current_avg * (total_requests - 1)) + response_time) / total_requests
                    self.history["models"][model]["average_response_time"] = new_avg
                    
                    # Calculate costs if available
                    if model in self.cost_rates:
                        input_cost = input_tokens * self.cost_rates[model]["input"]
                        output_cost = output_tokens * self.cost_rates[model]["output"]
                        self.history["models"][model]["total_cost"] += (input_cost + output_cost)
                    
                    self.save_history()
                    break
        except Exception as e:
            print(f"Error updating session: {e}")

    def end_session(self, session_id: str) -> None:
        """
        End a usage tracking session.
        
        Args:
            session_id (str): ID of the session to end
        """
        try:
            for session in self.history["sessions"]:
                if session["id"] == session_id:
                    session["end_time"] = datetime.now().isoformat()
                    self.save_history()
                    break
        except Exception as e:
            print(f"Error ending session: {e}")

    def get_usage_summary(self, days: int = 7) -> Dict:
        """
        Get usage summary for the specified time period.
        
        Args:
            days (int): Number of days to include in summary
            
        Returns:
            Dict: Summary of usage statistics
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            recent_sessions = [
                s for s in self.history["sessions"]
                if datetime.fromisoformat(s["start_time"]) > cutoff_date
            ]
            
            summary = {}
            for model in self.history["models"]:
                model_sessions = [s for s in recent_sessions if s["model"] == model]
                if model_sessions:
                    summary[model] = {
                        "total_tokens": sum(s["input_tokens"] + s["output_tokens"] 
                                         for s in model_sessions),
                        "total_requests": sum(s["requests"] for s in model_sessions),
                        "total_cost": self.calculate_cost(model, model_sessions),
                        "average_response_time": self.history["models"][model]["average_response_time"]
                    }
            return summary
        except Exception as e:
            print(f"Error getting usage summary: {e}")
            return {}

    def calculate_cost(self, model: str, sessions: List[Dict]) -> float:
        """
        Calculate total cost for given sessions.
        
        Args:
            model (str): Model name
            sessions (List[Dict]): List of sessions to calculate cost for
            
        Returns:
            float: Total cost for the sessions
        """
        if model not in self.cost_rates:
            return 0.0
        
        try:
            total_cost = 0.0
            for session in sessions:
                input_cost = session["input_tokens"] * self.cost_rates[model]["input"]
                output_cost = session["output_tokens"] * self.cost_rates[model]["output"]
                total_cost += input_cost + output_cost
            return total_cost
        except Exception as e:
            print(f"Error calculating cost: {e}")
            return 0.0

    def update_context_metrics(self, session_id: str) -> None:
        """
        Update context-related metrics for a session.
        
        Args:
            session_id (str): ID of the session to update
        """
        try:
            if self.context_manager:
                for session in self.history["sessions"]:
                    if session["id"] == session_id:
                        session["context_size"] = self.context_manager.get_context_size()
                        session["memory_usage"] = self.context_manager.get_memory_usage()
                        self.save_history()
                        break
        except Exception as e:
            print(f"Error updating context metrics: {e}")

    def record_rate_limit_hit(self, session_id: str) -> None:
        """
        Record a rate limit hit for a session.
        
        Args:
            session_id (str): ID of the session to update
        """
        try:
            for session in self.history["sessions"]:
                if session["id"] == session_id:
                    session["rate_limit_hits"] += 1
                    self.save_history()
                    break
        except Exception as e:
            print(f"Error recording rate limit hit: {e}")

    def update_remaining_capacity(self, session_id: str, capacity: int) -> None:
        """
        Update remaining capacity for a session.
        
        Args:
            session_id (str): ID of the session to update
            capacity (int): Remaining capacity value
        """
        try:
            for session in self.history["sessions"]:
                if session["id"] == session_id:
                    session["remaining_capacity"] = capacity
                    self.save_history()
                    break
        except Exception as e:
            print(f"Error updating remaining capacity: {e}")

    def record_time_between_requests(self, session_id: str, time_delta: float) -> None:
        """
        Record time between requests for a session.
        
        Args:
            session_id (str): ID of the session to update
            time_delta (float): Time between requests in seconds
        """
        try:
            for session in self.history["sessions"]:
                if session["id"] == session_id:
                    session["time_between_requests"].append(time_delta)
                    self.save_history()
                    break
        except Exception as e:
            print(f"Error recording time between requests: {e}")

    def record_content_length(self, session_id: str, length: int) -> None:
        """
        Record content length for a session.
        
        Args:
            session_id (str): ID of the session to update
            length (int): Length of content in characters
        """
        try:
            for session in self.history["sessions"]:
                if session["id"] == session_id:
                    session["content_lengths"].append(length)
                    self.save_history()
                    break
        except Exception as e:
            print(f"Error recording content length: {e}")

    def record_knowledge_access(self, session_id: str, pattern: str) -> None:
        """
        Record knowledge access pattern for a session.
        
        Args:
            session_id (str): ID of the session to update
            pattern (str): Knowledge access pattern description
        """
        try:
            for session in self.history["sessions"]:
                if session["id"] == session_id:
                    session["knowledge_access_patterns"].append(pattern)
                    self.save_history()
                    break
        except Exception as e:
            print(f"Error recording knowledge access pattern: {e}")

    def update_latency_breakdown(self, session_id: str, processing: float, 
                               network: float, total: float) -> None:
        """
        Update latency breakdown for a session.
        
        Args:
            session_id (str): ID of the session to update
            processing (float): Processing latency in seconds
            network (float): Network latency in seconds
            total (float): Total latency in seconds
        """
        try:
            for session in self.history["sessions"]:
                if session["id"] == session_id:
                    session["latency_breakdown"]["processing"] = processing
                    session["latency_breakdown"]["network"] = network
                    session["latency_breakdown"]["total"] = total
                    self.save_history()
                    break
        except Exception as e:
            print(f"Error updating latency breakdown: {e}")

    def analyze_uncertainty(self, session_id: str) -> Dict:
        """
        Analyze uncertainty levels for a session.
        
        Args:
            session_id (str): ID of the session to analyze
            
        Returns:
            Dict: Uncertainty analysis results
        """
        try:
            for session in self.history["sessions"]:
                if session["id"] == session_id:
                    if not session["uncertainty_levels"]:
                        return {}
                    
                    return {
                        "average": sum(session["uncertainty_levels"]) / len(session["uncertainty_levels"]),
                        "max": max(session["uncertainty_levels"]),
                        "min": min(session["uncertainty_levels"]),
                        "trend": self._calculate_trend(session["uncertainty_levels"])
                    }
            return {}
        except Exception as e:
            print(f"Error analyzing uncertainty: {e}")
            return {}

    def analyze_coherence(self, session_id: str) -> Dict:
        """
        Analyze response coherence for a session.
        
        Args:
            session_id (str): ID of the session to analyze
            
        Returns:
            Dict: Coherence analysis results
        """
        try:
            for session in self.history["sessions"]:
                if session["id"] == session_id:
                    if not session["coherence_scores"]:
                        return {}
                    
                    return {
                        "average": sum(session["coherence_scores"]) / len(session["coherence_scores"]),
                        "max": max(session["coherence_scores"]),
                        "min": min(session["coherence_scores"]),
                        "trend": self._calculate_trend(session["coherence_scores"])
                    }
            return {}
        except Exception as e:
            print(f"Error analyzing coherence: {e}")
            return {}

    def _calculate_trend(self, values: List[float]) -> float:
        """
        Calculate trend of a metric over time using linear regression.
        
        Args:
            values (List[float]): List of metric values
            
        Returns:
            float: Trend coefficient (positive = increasing, negative = decreasing)
        """
        try:
            n = len(values)
            if n < 2:
                return 0.0
                
            x = list(range(n))
            x_mean = sum(x) / n
            y_mean = sum(values) / n
            
            numerator = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(n))
            denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
            
            return numerator / denominator if denominator != 0 else 0.0
        except Exception as e:
            print(f"Error calculating trend: {e}")
            return 0.0

    def calculate_sleep_thresholds(self, session_id: str) -> Dict:
        """
        Calculate thresholds for triggering sleep state.
        
        Args:
            session_id (str): ID of the session to analyze
            
        Returns:
            Dict: Dictionary containing calculated thresholds
        """
        try:
            thresholds = {}
            for session in self.history["sessions"]:
                if session["id"] == session_id:
                    # Calculate token usage threshold
                    token_threshold = max(
                        1000,  # Minimum threshold
                        session["input_tokens"] + session["output_tokens"]
                    )
                    
                    # Calculate rate limit threshold
                    rate_threshold = max(
                        5,  # Minimum threshold
                        session["rate_limit_hits"] * 2
                    )
                    
                    # Calculate uncertainty threshold
                    uncertainty_analysis = self.analyze_uncertainty(session_id)
                    uncertainty_threshold = uncertainty_analysis.get("average", 0.5) * 1.2
                    
                    # Calculate coherence threshold
                    coherence_analysis = self.analyze_coherence(session_id)
                    coherence_threshold = coherence_analysis.get("average", 0.8) * 0.9
                    
                    thresholds = {
                        "token_threshold": token_threshold,
                        "rate_threshold": rate_threshold,
                        "uncertainty_threshold": uncertainty_threshold,
                        "coherence_threshold": coherence_threshold,
                        "context_size": session["context_size"],
                        "memory_usage": session["memory_usage"]
                    }
                    break
            return thresholds
        except Exception as e:
            print(f"Error calculating sleep thresholds: {e}")
            return {}

    def should_initiate_sleep(self, session_id: str) -> bool:
        """
        Determine if sleep state should be initiated.
        
        Args:
            session_id (str): ID of the session to check
            
        Returns:
            bool: True if sleep state should be initiated, False otherwise
        """
        try:
            thresholds = self.calculate_sleep_thresholds(session_id)
            if not thresholds:
                return False
                
            for session in self.history["sessions"]:
                if session["id"] == session_id:
                    # Check token usage
                    token_usage = session["input_tokens"] + session["output_tokens"]
                    if token_usage >= thresholds["token_threshold"]:
                        return True
                        
                    # Check rate limit hits
                    if session["rate_limit_hits"] >= thresholds["rate_threshold"]:
                        return True
                        
                    # Check uncertainty levels
                    uncertainty_analysis = self.analyze_uncertainty(session_id)
                    if uncertainty_analysis.get("average", 0) >= thresholds["uncertainty_threshold"]:
                        return True
                        
                    # Check coherence scores
                    coherence_analysis = self.analyze_coherence(session_id)
                    if coherence_analysis.get("average", 1) <= thresholds["coherence_threshold"]:
                        return True
                        
                    # Check context size
                    if session["context_size"] >= thresholds.get("context_size", 0.8):
                        return True
                        
                    # Check memory usage
                    if session["memory_usage"] >= thresholds.get("memory_usage", 0.8):
                        return True
                        
                    break
            return False
        except Exception as e:
            print(f"Error checking sleep conditions: {e}")
            return False

    def get_usage_trends(self, days: int = 30) -> pd.DataFrame:
        """
        Get usage trends for the specified time period.
        
        Args:
            days (int): Number of days to include in trends
            
        Returns:
            pd.DataFrame: DataFrame containing usage trends
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            recent_sessions = [
                s for s in self.history["sessions"]
                if datetime.fromisoformat(s["start_time"]) > cutoff_date
            ]
            
            daily_usage = {}
            for session in recent_sessions:
                date = datetime.fromisoformat(session["start_time"]).date()
                model = session["model"]
                
                if date not in daily_usage:
                    daily_usage[date] = {}
                if model not in daily_usage[date]:
                    daily_usage[date][model] = {
                        "tokens": 0,
                        "requests": 0,
                        "cost": 0.0
                    }
                
                daily_usage[date][model]["tokens"] += (
                    session["input_tokens"] + session["output_tokens"]
                )
                daily_usage[date][model]["requests"] += session["requests"]
                daily_usage[date][model]["cost"] += self.calculate_cost(
                    model, [session]
                )
            
            records = []
            for date, models in daily_usage.items():
                for model, stats in models.items():
                    records.append({
                        "date": date,
                        "model": model,
                        "tokens": stats["tokens"],
                        "requests": stats["requests"],
                        "cost": stats["cost"]
                    })
            
            return pd.DataFrame(records)
        except Exception as e:
            print(f"Error getting usage trends: {e}")
            return pd.DataFrame()
