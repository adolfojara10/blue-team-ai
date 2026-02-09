# File: /final-filter/final-filter/src/resource_monitor.py
import threading
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import os

# CPU monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# GPU monitoring (NVIDIA)
try:
    import pynvml
    pynvml.nvmlInit()
    PYNVML_AVAILABLE = True
except (ImportError, Exception):
    PYNVML_AVAILABLE = False

# Alternative GPU monitoring
try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False


@dataclass
class ResourceSample:
    """Single sample of resource usage"""
    timestamp: float
    cpu_percent: float = 0.0
    cpu_percent_per_core: List[float] = field(default_factory=list)
    memory_percent: float = 0.0
    memory_used_mb: float = 0.0
    memory_available_mb: float = 0.0
    gpu_percent: float = 0.0
    gpu_memory_percent: float = 0.0
    gpu_memory_used_mb: float = 0.0
    gpu_memory_total_mb: float = 0.0
    gpu_temperature: float = 0.0
    gpu_power_watts: float = 0.0


@dataclass
class ResourceStats:
    """Statistics from monitoring session"""
    # CPU stats
    cpu_percent_avg: float = 0.0
    cpu_percent_max: float = 0.0
    cpu_percent_min: float = 0.0
    memory_percent_avg: float = 0.0
    memory_percent_max: float = 0.0
    memory_used_mb_avg: float = 0.0
    memory_used_mb_max: float = 0.0
    
    # GPU stats
    gpu_percent_avg: float = 0.0
    gpu_percent_max: float = 0.0
    gpu_percent_min: float = 0.0
    gpu_memory_percent_avg: float = 0.0
    gpu_memory_percent_max: float = 0.0
    gpu_memory_used_mb_avg: float = 0.0
    gpu_memory_used_mb_max: float = 0.0
    gpu_temperature_avg: float = 0.0
    gpu_temperature_max: float = 0.0
    gpu_power_watts_avg: float = 0.0
    gpu_power_watts_max: float = 0.0
    
    # Metadata
    num_samples: int = 0
    monitoring_duration_sec: float = 0.0
    gpu_available: bool = False
    gpu_name: str = ""
    cpu_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for CSV logging"""
        return {
            'cpu_percent_avg': round(self.cpu_percent_avg, 2),
            'cpu_percent_max': round(self.cpu_percent_max, 2),
            'cpu_percent_min': round(self.cpu_percent_min, 2),
            'memory_percent_avg': round(self.memory_percent_avg, 2),
            'memory_percent_max': round(self.memory_percent_max, 2),
            'memory_used_mb_avg': round(self.memory_used_mb_avg, 2),
            'memory_used_mb_max': round(self.memory_used_mb_max, 2),
            'gpu_percent_avg': round(self.gpu_percent_avg, 2),
            'gpu_percent_max': round(self.gpu_percent_max, 2),
            'gpu_percent_min': round(self.gpu_percent_min, 2),
            'gpu_memory_percent_avg': round(self.gpu_memory_percent_avg, 2),
            'gpu_memory_percent_max': round(self.gpu_memory_percent_max, 2),
            'gpu_memory_used_mb_avg': round(self.gpu_memory_used_mb_avg, 2),
            'gpu_memory_used_mb_max': round(self.gpu_memory_used_mb_max, 2),
            'gpu_temperature_avg': round(self.gpu_temperature_avg, 2),
            'gpu_temperature_max': round(self.gpu_temperature_max, 2),
            'gpu_power_watts_avg': round(self.gpu_power_watts_avg, 2),
            'gpu_power_watts_max': round(self.gpu_power_watts_max, 2),
            'num_resource_samples': self.num_samples,
            'monitoring_duration_sec': round(self.monitoring_duration_sec, 4),
            'gpu_available': self.gpu_available,
            'gpu_name': self.gpu_name,
            'cpu_count': self.cpu_count
        }


class ResourceMonitor:
    """
    Monitor CPU and GPU resource usage during operations.
    Uses background thread for continuous sampling.
    """
    
    def __init__(self, 
                 sample_interval: float = 0.1,  # Sample every 100ms
                 gpu_index: int = 0,
                 logger=None):
        """
        Initialize resource monitor.
        
        Args:
            sample_interval: Time between samples in seconds
            gpu_index: GPU device index to monitor
            logger: Optional logger instance
        """
        self.sample_interval = sample_interval
        self.gpu_index = gpu_index
        self.logger = logger
        
        self._samples: List[ResourceSample] = []
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._start_time: float = 0.0
        
        # Initialize process handle for CPU monitoring
        self._process = psutil.Process(os.getpid()) if PSUTIL_AVAILABLE else None
        
        # Initialize GPU handle
        self._gpu_handle = None
        self._gpu_name = ""
        self._init_gpu()
        
        # Log availability
        if self.logger:
            self.logger.info(f"📊 Resource Monitor initialized:")
            self.logger.info(f"   CPU monitoring: {'✅ Available' if PSUTIL_AVAILABLE else '❌ Not available'}")
            self.logger.info(f"   GPU monitoring: {'✅ Available' if self._gpu_handle else '❌ Not available'}")
            if self._gpu_name:
                self.logger.info(f"   GPU: {self._gpu_name}")
    
    def _init_gpu(self):
        """Initialize GPU monitoring"""
        if PYNVML_AVAILABLE:
            try:
                device_count = pynvml.nvmlDeviceGetCount()
                if device_count > self.gpu_index:
                    self._gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
                    self._gpu_name = pynvml.nvmlDeviceGetName(self._gpu_handle)
                    if isinstance(self._gpu_name, bytes):
                        self._gpu_name = self._gpu_name.decode('utf-8')
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"⚠️ Failed to initialize GPU monitoring: {e}")
    
    def _get_cpu_metrics(self) -> Dict[str, Any]:
        """Get current CPU metrics"""
        metrics = {
            'cpu_percent': 0.0,
            'cpu_percent_per_core': [],
            'memory_percent': 0.0,
            'memory_used_mb': 0.0,
            'memory_available_mb': 0.0
        }
        
        if not PSUTIL_AVAILABLE:
            return metrics
        
        try:
            # System-wide CPU usage
            metrics['cpu_percent'] = psutil.cpu_percent(interval=None)
            metrics['cpu_percent_per_core'] = psutil.cpu_percent(interval=None, percpu=True)
            
            # Memory usage
            memory = psutil.virtual_memory()
            metrics['memory_percent'] = memory.percent
            metrics['memory_used_mb'] = memory.used / (1024 * 1024)
            metrics['memory_available_mb'] = memory.available / (1024 * 1024)
            
            # Process-specific CPU (optional, can be added)
            # process_cpu = self._process.cpu_percent(interval=None)
            
        except Exception as e:
            if self.logger:
                self.logger.debug(f"Error getting CPU metrics: {e}")
        
        return metrics
    
    def _get_gpu_metrics(self) -> Dict[str, Any]:
        """Get current GPU metrics"""
        metrics = {
            'gpu_percent': 0.0,
            'gpu_memory_percent': 0.0,
            'gpu_memory_used_mb': 0.0,
            'gpu_memory_total_mb': 0.0,
            'gpu_temperature': 0.0,
            'gpu_power_watts': 0.0
        }
        
        if not self._gpu_handle:
            # Try GPUtil as fallback
            if GPUTIL_AVAILABLE:
                try:
                    gpus = GPUtil.getGPUs()
                    if gpus and len(gpus) > self.gpu_index:
                        gpu = gpus[self.gpu_index]
                        metrics['gpu_percent'] = gpu.load * 100
                        metrics['gpu_memory_percent'] = gpu.memoryUtil * 100
                        metrics['gpu_memory_used_mb'] = gpu.memoryUsed
                        metrics['gpu_memory_total_mb'] = gpu.memoryTotal
                        metrics['gpu_temperature'] = gpu.temperature
                except Exception:
                    pass
            return metrics
        
        try:
            # GPU utilization
            utilization = pynvml.nvmlDeviceGetUtilizationRates(self._gpu_handle)
            metrics['gpu_percent'] = utilization.gpu
            
            # Memory info
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(self._gpu_handle)
            metrics['gpu_memory_used_mb'] = memory_info.used / (1024 * 1024)
            metrics['gpu_memory_total_mb'] = memory_info.total / (1024 * 1024)
            metrics['gpu_memory_percent'] = (memory_info.used / memory_info.total) * 100
            
            # Temperature
            try:
                metrics['gpu_temperature'] = pynvml.nvmlDeviceGetTemperature(
                    self._gpu_handle, pynvml.NVML_TEMPERATURE_GPU
                )
            except Exception:
                pass
            
            # Power usage
            try:
                power_mw = pynvml.nvmlDeviceGetPowerUsage(self._gpu_handle)
                metrics['gpu_power_watts'] = power_mw / 1000.0
            except Exception:
                pass
                
        except Exception as e:
            if self.logger:
                self.logger.debug(f"Error getting GPU metrics: {e}")
        
        return metrics
    
    def _sample(self) -> ResourceSample:
        """Take a single sample of all metrics"""
        cpu_metrics = self._get_cpu_metrics()
        gpu_metrics = self._get_gpu_metrics()
        
        return ResourceSample(
            timestamp=time.time(),
            **cpu_metrics,
            **gpu_metrics
        )
    
    def _monitor_loop(self):
        """Background monitoring loop"""
        # Initial CPU call to initialize (psutil needs this)
        if PSUTIL_AVAILABLE:
            psutil.cpu_percent(interval=None)
        
        while self._monitoring:
            sample = self._sample()
            
            with self._lock:
                self._samples.append(sample)
            
            time.sleep(self.sample_interval)
    
    def start(self):
        """Start monitoring in background thread"""
        if self._monitoring:
            return
        
        self._samples = []
        self._monitoring = True
        self._start_time = time.time()
        
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        
        if self.logger:
            self.logger.debug("📊 Resource monitoring started")
    
    def stop(self) -> ResourceStats:
        """Stop monitoring and return statistics"""
        self._monitoring = False
        
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)
            self._monitor_thread = None
        
        # Calculate statistics
        stats = self._calculate_stats()
        
        if self.logger:
            self.logger.debug(f"📊 Resource monitoring stopped. {stats.num_samples} samples collected.")
        
        return stats
    
    def _calculate_stats(self) -> ResourceStats:
        """Calculate statistics from collected samples"""
        with self._lock:
            samples = list(self._samples)
        
        if not samples:
            return ResourceStats(
                gpu_available=self._gpu_handle is not None or GPUTIL_AVAILABLE,
                gpu_name=self._gpu_name,
                cpu_count=psutil.cpu_count() if PSUTIL_AVAILABLE else 0
            )
        
        # Extract values
        cpu_percents = [s.cpu_percent for s in samples]
        memory_percents = [s.memory_percent for s in samples]
        memory_used = [s.memory_used_mb for s in samples]
        gpu_percents = [s.gpu_percent for s in samples]
        gpu_memory_percents = [s.gpu_memory_percent for s in samples]
        gpu_memory_used = [s.gpu_memory_used_mb for s in samples]
        gpu_temps = [s.gpu_temperature for s in samples if s.gpu_temperature > 0]
        gpu_powers = [s.gpu_power_watts for s in samples if s.gpu_power_watts > 0]
        
        duration = samples[-1].timestamp - samples[0].timestamp if len(samples) > 1 else 0
        
        return ResourceStats(
            # CPU stats
            cpu_percent_avg=sum(cpu_percents) / len(cpu_percents) if cpu_percents else 0,
            cpu_percent_max=max(cpu_percents) if cpu_percents else 0,
            cpu_percent_min=min(cpu_percents) if cpu_percents else 0,
            memory_percent_avg=sum(memory_percents) / len(memory_percents) if memory_percents else 0,
            memory_percent_max=max(memory_percents) if memory_percents else 0,
            memory_used_mb_avg=sum(memory_used) / len(memory_used) if memory_used else 0,
            memory_used_mb_max=max(memory_used) if memory_used else 0,
            
            # GPU stats
            gpu_percent_avg=sum(gpu_percents) / len(gpu_percents) if gpu_percents else 0,
            gpu_percent_max=max(gpu_percents) if gpu_percents else 0,
            gpu_percent_min=min(gpu_percents) if gpu_percents else 0,
            gpu_memory_percent_avg=sum(gpu_memory_percents) / len(gpu_memory_percents) if gpu_memory_percents else 0,
            gpu_memory_percent_max=max(gpu_memory_percents) if gpu_memory_percents else 0,
            gpu_memory_used_mb_avg=sum(gpu_memory_used) / len(gpu_memory_used) if gpu_memory_used else 0,
            gpu_memory_used_mb_max=max(gpu_memory_used) if gpu_memory_used else 0,
            gpu_temperature_avg=sum(gpu_temps) / len(gpu_temps) if gpu_temps else 0,
            gpu_temperature_max=max(gpu_temps) if gpu_temps else 0,
            gpu_power_watts_avg=sum(gpu_powers) / len(gpu_powers) if gpu_powers else 0,
            gpu_power_watts_max=max(gpu_powers) if gpu_powers else 0,
            
            # Metadata
            num_samples=len(samples),
            monitoring_duration_sec=duration,
            gpu_available=self._gpu_handle is not None or GPUTIL_AVAILABLE,
            gpu_name=self._gpu_name,
            cpu_count=psutil.cpu_count() if PSUTIL_AVAILABLE else 0
        )
    
    def get_instant_metrics(self) -> Dict[str, Any]:
        """Get instant snapshot of current resource usage (no background monitoring)"""
        cpu_metrics = self._get_cpu_metrics()
        gpu_metrics = self._get_gpu_metrics()
        
        return {
            **cpu_metrics,
            **gpu_metrics,
            'gpu_available': self._gpu_handle is not None or GPUTIL_AVAILABLE,
            'gpu_name': self._gpu_name,
            'cpu_count': psutil.cpu_count() if PSUTIL_AVAILABLE else 0
        }


class ResourceMonitorContext:
    """Context manager for easy resource monitoring"""
    
    def __init__(self, monitor: ResourceMonitor):
        self.monitor = monitor
        self.stats: Optional[ResourceStats] = None
    
    def __enter__(self):
        self.monitor.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stats = self.monitor.stop()
        return False
    
    def get_stats(self) -> ResourceStats:
        return self.stats


# Convenience function
def monitor_resources(func):
    """Decorator to monitor resources during function execution"""
    def wrapper(*args, **kwargs):
        monitor = ResourceMonitor()
        monitor.start()
        try:
            result = func(*args, **kwargs)
            return result, monitor.stop()
        finally:
            if monitor._monitoring:
                monitor.stop()
    return wrapper