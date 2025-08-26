import time
import psutil
import torch
import numpy as np
from typing import Dict, List, Callable, Any
from dataclasses import dataclass
from contextlib import contextmanager

@dataclass
class BenchmarkResult:
    """Container for benchmark results"""
    name: str
    execution_time: float
    memory_used_mb: float
    gpu_memory_used_mb: float
    accuracy: float = None
    loss: float = None
    additional_metrics: Dict[str, Any] = None

class ModelBenchmark:
    """Comprehensive benchmarking utility for ML models"""
    
    def __init__(self):
        self.results: List[BenchmarkResult] = []
        
    @contextmanager
    def measure_resources(self, name: str):
        """Context manager to measure execution time and memory usage"""
        process = psutil.Process()
        start_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            start_gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
        else:
            start_gpu_memory = 0
            
        start_time = time.time()
        
        try:
            yield
        finally:
            end_time = time.time()
            end_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            if torch.cuda.is_available():
                end_gpu_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
                gpu_memory_used = end_gpu_memory - start_gpu_memory
            else:
                gpu_memory_used = 0
                
            execution_time = end_time - start_time
            memory_used = end_memory - start_memory
            
            result = BenchmarkResult(
                name=name,
                execution_time=execution_time,
                memory_used_mb=memory_used,
                gpu_memory_used_mb=gpu_memory_used
            )
            self.results.append(result)
    
    def benchmark_model(self, model_fn: Callable, data_loader, name: str, 
                       criterion=None, device='cpu') -> BenchmarkResult:
        """Benchmark a model's training or inference performance"""
        with self.measure_resources(name):
            model = model_fn()
            model.to(device)
            
            total_loss = 0
            correct = 0
            total = 0
            
            for batch_idx, (data, target) in enumerate(data_loader):
                data, target = data.to(device), target.to(device)
                output = model(data)
                
                if criterion:
                    loss = criterion(output, target)
                    total_loss += loss.item()
                
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        # Update the latest result with accuracy and loss
        latest_result = self.results[-1]
        latest_result.accuracy = correct / total if total > 0 else 0
        latest_result.loss = total_loss / len(data_loader) if criterion else None
        
        return latest_result
    
    def compare_models(self, model_configs: List[Dict]) -> None:
        """Compare multiple models and print results"""
        print(f"{'Model Name':<20} {'Time (s)':<10} {'Memory (MB)':<12} {'GPU Mem (MB)':<12} {'Accuracy':<10}")
        print("-" * 70)
        
        for result in self.results:
            print(f"{result.name:<20} {result.execution_time:<10.3f} "
                  f"{result.memory_used_mb:<12.2f} {result.gpu_memory_used_mb:<12.2f} "
                  f"{result.accuracy:<10.3f}" if result.accuracy else "N/A")
    
    def get_summary_stats(self) -> Dict[str, float]:
        """Get summary statistics across all benchmarks"""
        if not self.results:
            return {}
            
        times = [r.execution_time for r in self.results]
        memories = [r.memory_used_mb for r in self.results]
        
        return {
            'avg_time': np.mean(times),
            'std_time': np.std(times),
            'avg_memory': np.mean(memories),
            'std_memory': np.std(memories)
        }