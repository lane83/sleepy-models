def __init__(self, data_dir: str = "data"):
    """
    Initialize the performance tracker.
    
    Args:
        data_dir (str): Directory to store performance data
    """
    self.data_dir = data_dir
    os.makedirs(data_dir, exist_ok=True)
    
    self.performance_file = os.path.join(data_dir, "model_performance.json")
    self.benchmark_file = os.path.join(data_dir, "benchmark_results.json")
    self.lock_file = os.path.join(data_dir, "model_performance.lock")
    self.file_lock = FileLock(self.lock_file)
    
    # Performance thresholds
    self.thresholds = {
        "max_response_time": 5.0,  # seconds
        "min_token_efficiency": 10.0,  # tokens per second
        "max_error_rate": 0.05,  # 5%
        "max_cost_per_token": 0.0002  # $0.0002 per token
    }
    
    self.load_performance_data()

def load_performance_data(self) -> None:
    """Load performance data from files with proper error handling."""
    with self.file_lock:
        try:
            if os.path.exists(self.performance_file):
                with open(self.performance_file, 'r') as f:
                    self.performance_data = json.load(f)
            else:
                self.performance_data = {}
                
            if os.path.exists(self.benchmark_file):
                with open(self.benchmark_file, 'r') as f:
                    self.benchmark_data = json.load(f)
            else:
                self.benchmark_data = {
                    "benchmarks": [],
                    "results": {}
                }
        except Exception as e:
            print(f"Error loading performance data: {e}")
            self.performance_data = {}
            self.benchmark_data = {
                "benchmarks": [],
                "results": {}
            }

def save_performance_data(self) -> None:
    """Save performance data with atomic write operations."""
    with self.file_lock:
        try:
            # Save performance data
            temp_perf = f"{self.performance_file}.tmp"
            with open(temp_perf, 'w') as f:
                json.dump(self.performance_data, f)
            os.replace(temp_perf, self.performance_file)

            # Save benchmark data
            temp_bench = f"{self.benchmark_file}.tmp"
            with open(temp_bench, 'w') as f:
                json.dump(self.benchmark_data, f)
            os.replace(temp_bench, self.benchmark_file)
        except Exception as e:
            print(f"Error saving performance data: {e}")
            for temp_file in [temp_perf, temp_bench]:
                if os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                    except:
                        pass

def record_completion(self, model: str, start_time: float, 
                     input_tokens: int, output_tokens: int,
                     success: bool, cost: float) -> None:
    """
    Record performance metrics for a model completion.
    
    Args:
        model (str): Name of the model
        start_time (float): Start time of the request
        input_tokens (int): Number of input tokens
        output_tokens (int): Number of output tokens
        success (bool): Whether the completion was successful
        cost (float): Cost of the completion
    """
    try:
        end_time = time.time()
        response_time = end_time - start_time
        total_tokens = input_tokens + output_tokens
        
        if model not in self.performance_data:
            self.performance_data[model] = []
            
        metrics = PerformanceMetrics(
            response_time=response_time,
            token_efficiency=total_tokens / response_time if response_time > 0 else 0,
            error_rate=0.0,
            cost_per_token=cost / total_tokens if total_tokens > 0 else 0,
            successful_completions=1 if success else 0,
            failed_completions=0 if success else 1,
            average_tokens_per_request=total_tokens,
            timestamp=datetime.now().isoformat()
        )
        
        self.performance_data[model].append(asdict(metrics))
        self.save_performance_data()
    except Exception as e:
        print(f"Error recording completion: {e}")

def get_model_performance_summary(self, model: str, 
                                last_n_requests: Optional[int] = None) -> Optional[Dict]:
    """
    Get performance summary for a specific model.
    
    Args:
        model (str): Model name
        last_n_requests (Optional[int]): Number of recent requests to include
        
    Returns:
        Optional[Dict]: Performance summary or None if no data available
    """
    try:
        if model not in self.performance_data:
            return None
            
        data = self.performance_data[model]
        if last_n_requests:
            data = data[-last_n_requests:]
            
        total_requests = len(data)
        if total_requests == 0:
            return None
            
        successful = sum(d["successful_completions"] for d in data)
        failed = sum(d["failed_completions"] for d in data)
        
        return {
            "average_response_time": np.mean([d["response_time"] for d in data]),
            "token_efficiency": np.mean([d["token_efficiency"] for d in data]),
            "error_rate": failed / total_requests if total_requests > 0 else 0,
            "average_cost_per_token": np.mean([d["cost_per_token"] for d in data]),
            "success_rate": successful / total_requests if total_requests > 0 else 0,
            "average_tokens_per_request": np.mean([d["average_tokens_per_request"] for d in data]),
            "total_requests": total_requests
        }
    except Exception as e:
        print(f"Error getting performance summary: {e}")
        return None

def check_performance_issues(self, model: str) -> List[str]:
    """
    Check for performance issues based on thresholds.
    
    Args:
        model (str): Model to check
        
    Returns:
        List[str]: List of identified issues
    """
    issues = []
    summary = self.get_model_performance_summary(model, last_n_requests=100)
    
    if summary:
        if summary["average_response_time"] > self.thresholds["max_response_time"]:
            issues.append("High response time")
        if summary["token_efficiency"] < self.thresholds["min_token_efficiency"]:
            issues.append("Low token efficiency")
        if summary["error_rate"] > self.thresholds["max_error_rate"]:
            issues.append("High error rate")
        if summary["average_cost_per_token"] > self.thresholds["max_cost_per_token"]:
            issues.append("High cost per token")
    
    return issues

def run_benchmark(self, models: List[str], benchmark_suite: List[Dict]) -> str:
    """
    Run a benchmark suite across multiple models.
    
    Args:
        models (List[str]): List of models to benchmark
        benchmark_suite (List[Dict]): List of test cases
        
    Returns:
        str: Benchmark ID
    """
    try:
        benchmark_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.benchmark_data["benchmarks"].append({
            "id": benchmark_id,
            "timestamp": datetime.now().isoformat(),
            "models": models,
            "suite_size": len(benchmark_suite)
        })
        
        self.benchmark_data["results"][benchmark_id] = {
            model: {
                "prompts": [],
                "metrics": {
                    "average_response_time": 0,
                    "token_efficiency": 0,
                    "error_rate": 0,
                    "cost_efficiency": 0,
                    "accuracy": 0
                }
            } for model in models
        }
        
        self.save_performance_data()
        return benchmark_id
    except Exception as e:
        print(f"Error running benchmark: {e}")
        return ""

def get_benchmark_results(self, benchmark_id: str) -> Optional[Dict]:
    """
    Get results for a specific benchmark.
    
    Args:
        benchmark_id (str): ID of the benchmark
        
    Returns:
        Optional[Dict]: Benchmark results or None if not found
    """
    try:
        if benchmark_id not in self.benchmark_data["results"]:
            return None
        return self.benchmark_data["results"][benchmark_id]
    except Exception as e:
        print(f"Error getting benchmark results: {e}")
        return None

def cleanup_old_data(self, days: int = 30) -> None:
    """
    Clean up performance data older than specified days.
    
    Args:
        days (int): Number of days of data to keep
    """
    try:
        cutoff_date = datetime.now() - timedelta(days=days)
        
        for model in self.performance_data:
            self.performance_data[model] = [
                entry for entry in self.performance_data[model]
                if datetime.fromisoformat(entry["timestamp"]) > cutoff_date
            ]
        
        self.save_performance_data()
    except Exception as e:
        print(f"Error cleaning up old data: {e}")