#!/usr/bin/env python3
"""
Training monitoring and logging utilities for Llama 3.1 Instruct 8B training.
Provides real-time monitoring, metrics tracking, and automated alerts.
"""

import os
import sys
import time
import json
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import psutil
import GPUtil
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

class TrainingMonitor:
    """Monitor training progress and system resources."""
    
    def __init__(self, 
                 checkpoint_dir: str,
                 tensorboard_dir: str,
                 log_file: Optional[str] = None):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.tensorboard_dir = Path(tensorboard_dir)
        self.log_file = log_file
        self.start_time = datetime.now()
        
        # Create monitoring directory
        self.monitor_dir = Path("./monitoring")
        self.monitor_dir.mkdir(exist_ok=True)
        
        # Initialize metrics storage
        self.metrics_history = []
        self.system_metrics = []
        
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system resource usage."""
        
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_used_gb': psutil.virtual_memory().used / (1024**3),
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
            'disk_usage_percent': psutil.disk_usage('/').percent,
        }
        
        # GPU metrics
        try:
            gpus = GPUtil.getGPUs()
            gpu_metrics = []
            for i, gpu in enumerate(gpus):
                gpu_metrics.append({
                    'gpu_id': i,
                    'gpu_name': gpu.name,
                    'gpu_load': gpu.load * 100,
                    'gpu_memory_used': gpu.memoryUsed,
                    'gpu_memory_total': gpu.memoryTotal,
                    'gpu_memory_percent': (gpu.memoryUsed / gpu.memoryTotal) * 100,
                    'gpu_temperature': gpu.temperature
                })
            metrics['gpus'] = gpu_metrics
        except Exception as e:
            metrics['gpu_error'] = str(e)
            
        return metrics
    
    def parse_tensorboard_logs(self) -> Dict[str, List]:
        """Parse TensorBoard logs to extract training metrics."""
        
        metrics = {
            'steps': [],
            'loss': [],
            'learning_rate': [],
            'throughput': [],
            'timestamps': []
        }
        
        try:
            # Find the latest tensorboard log directory
            tb_dirs = [d for d in self.tensorboard_dir.iterdir() if d.is_dir()]
            if not tb_dirs:
                return metrics
                
            latest_dir = max(tb_dirs, key=os.path.getctime)
            
            # Load tensorboard data
            ea = EventAccumulator(str(latest_dir))
            ea.Reload()
            
            # Extract scalar metrics
            scalar_tags = ea.Tags()['scalars']
            
            for tag in scalar_tags:
                if 'loss' in tag.lower():
                    events = ea.Scalars(tag)
                    for event in events:
                        if len(metrics['steps']) == 0 or event.step not in metrics['steps']:
                            metrics['steps'].append(event.step)
                            metrics['loss'].append(event.value)
                            metrics['timestamps'].append(datetime.fromtimestamp(event.wall_time))
                
                elif 'learning_rate' in tag.lower() or 'lr' in tag.lower():
                    events = ea.Scalars(tag)
                    for event in events:
                        if event.step in metrics['steps']:
                            idx = metrics['steps'].index(event.step)
                            if len(metrics['learning_rate']) <= idx:
                                metrics['learning_rate'].extend([0] * (idx + 1 - len(metrics['learning_rate'])))
                            metrics['learning_rate'][idx] = event.value
                
                elif 'throughput' in tag.lower() or 'tokens_per_sec' in tag.lower():
                    events = ea.Scalars(tag)
                    for event in events:
                        if event.step in metrics['steps']:
                            idx = metrics['steps'].index(event.step)
                            if len(metrics['throughput']) <= idx:
                                metrics['throughput'].extend([0] * (idx + 1 - len(metrics['throughput'])))
                            metrics['throughput'][idx] = event.value
                            
        except Exception as e:
            print(f"Error parsing tensorboard logs: {e}")
            
        return metrics
    
    def get_checkpoint_info(self) -> Dict[str, Any]:
        """Get information about saved checkpoints."""
        
        info = {
            'checkpoint_count': 0,
            'latest_checkpoint': None,
            'checkpoint_sizes': [],
            'total_size_gb': 0
        }
        
        if not self.checkpoint_dir.exists():
            return info
            
        checkpoints = list(self.checkpoint_dir.glob('iter_*'))
        info['checkpoint_count'] = len(checkpoints)
        
        if checkpoints:
            latest = max(checkpoints, key=os.path.getctime)
            info['latest_checkpoint'] = latest.name
            
            # Calculate sizes
            total_size = 0
            for cp in checkpoints:
                if cp.is_dir():
                    size = sum(f.stat().st_size for f in cp.rglob('*') if f.is_file())
                else:
                    size = cp.stat().st_size
                info['checkpoint_sizes'].append(size / (1024**3))  # GB
                total_size += size
                
            info['total_size_gb'] = total_size / (1024**3)
            
        return info
    
    def generate_report(self) -> str:
        """Generate a comprehensive training report."""
        
        # Get current metrics
        system_metrics = self.get_system_metrics()
        tb_metrics = self.parse_tensorboard_logs()
        checkpoint_info = self.get_checkpoint_info()
        
        # Calculate training duration
        duration = datetime.now() - self.start_time
        
        report = f"""
# Llama 3.1 Instruct 8B Training Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Training Overview
- **Start Time**: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}
- **Duration**: {str(duration).split('.')[0]}
- **Status**: {'Running' if self.is_training_active() else 'Stopped'}

## Training Progress
- **Total Steps**: {len(tb_metrics['steps'])}
- **Latest Step**: {max(tb_metrics['steps']) if tb_metrics['steps'] else 'N/A'}
- **Current Loss**: {tb_metrics['loss'][-1]:.4f if tb_metrics['loss'] else 'N/A'}
- **Learning Rate**: {tb_metrics['learning_rate'][-1]:.2e if tb_metrics['learning_rate'] else 'N/A'}
- **Throughput**: {tb_metrics['throughput'][-1]:.1f if tb_metrics['throughput'] else 'N/A'} tokens/sec

## System Resources
- **CPU Usage**: {system_metrics['cpu_percent']:.1f}%
- **Memory Usage**: {system_metrics['memory_percent']:.1f}% ({system_metrics['memory_used_gb']:.1f}GB / {system_metrics['memory_total_gb']:.1f}GB)
- **Disk Usage**: {system_metrics['disk_usage_percent']:.1f}%

## GPU Status
"""
        
        if 'gpus' in system_metrics:
            for gpu in system_metrics['gpus']:
                report += f"""
- **GPU {gpu['gpu_id']} ({gpu['gpu_name']})**:
  - Load: {gpu['gpu_load']:.1f}%
  - Memory: {gpu['gpu_memory_percent']:.1f}% ({gpu['gpu_memory_used']}MB / {gpu['gpu_memory_total']}MB)
  - Temperature: {gpu['gpu_temperature']}°C
"""
        else:
            report += "- No GPU information available\n"
        
        report += f"""
## Checkpoints
- **Count**: {checkpoint_info['checkpoint_count']}
- **Latest**: {checkpoint_info['latest_checkpoint'] or 'None'}
- **Total Size**: {checkpoint_info['total_size_gb']:.2f}GB

## Recommendations
"""
        
        # Add recommendations based on metrics
        recommendations = self.get_recommendations(system_metrics, tb_metrics)
        for rec in recommendations:
            report += f"- {rec}\n"
            
        return report
    
    def get_recommendations(self, system_metrics: Dict, tb_metrics: Dict) -> List[str]:
        """Generate recommendations based on current metrics."""
        
        recommendations = []
        
        # GPU utilization
        if 'gpus' in system_metrics:
            avg_gpu_load = sum(gpu['gpu_load'] for gpu in system_metrics['gpus']) / len(system_metrics['gpus'])
            if avg_gpu_load < 70:
                recommendations.append("GPU utilization is low. Consider increasing batch size or reducing model parallelism.")
            elif avg_gpu_load > 95:
                recommendations.append("GPU utilization is very high. Monitor for potential bottlenecks.")
        
        # Memory usage
        if system_metrics['memory_percent'] > 90:
            recommendations.append("System memory usage is high. Consider reducing batch size or enabling gradient checkpointing.")
        
        # Loss trends
        if len(tb_metrics['loss']) > 10:
            recent_loss = tb_metrics['loss'][-5:]
            if len(set(recent_loss)) == 1:  # Loss not changing
                recommendations.append("Loss appears to have plateaued. Consider adjusting learning rate or checking data quality.")
            elif recent_loss[-1] > recent_loss[0]:  # Loss increasing
                recommendations.append("Loss is increasing. Check for gradient explosion or learning rate issues.")
        
        # Throughput
        if tb_metrics['throughput'] and tb_metrics['throughput'][-1] < 1000:
            recommendations.append("Training throughput is low. Consider optimizing data loading or model parallelism.")
        
        return recommendations
    
    def is_training_active(self) -> bool:
        """Check if training process is currently active."""
        
        try:
            # Look for python processes running pretrain_gpt.py
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                if proc.info['cmdline'] and any('pretrain_gpt.py' in arg for arg in proc.info['cmdline']):
                    return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
            
        return False
    
    def create_plots(self):
        """Create training progress plots."""
        
        tb_metrics = self.parse_tensorboard_logs()
        
        if not tb_metrics['steps']:
            print("No training data available for plotting")
            return
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Llama 3.1 Instruct 8B Training Progress', fontsize=16)
        
        # Loss plot
        if tb_metrics['loss']:
            axes[0, 0].plot(tb_metrics['steps'], tb_metrics['loss'], 'b-', linewidth=2)
            axes[0, 0].set_title('Training Loss')
            axes[0, 0].set_xlabel('Steps')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].grid(True, alpha=0.3)
        
        # Learning rate plot
        if tb_metrics['learning_rate']:
            axes[0, 1].plot(tb_metrics['steps'], tb_metrics['learning_rate'], 'r-', linewidth=2)
            axes[0, 1].set_title('Learning Rate')
            axes[0, 1].set_xlabel('Steps')
            axes[0, 1].set_ylabel('Learning Rate')
            axes[0, 1].set_yscale('log')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Throughput plot
        if tb_metrics['throughput']:
            axes[1, 0].plot(tb_metrics['steps'], tb_metrics['throughput'], 'g-', linewidth=2)
            axes[1, 0].set_title('Training Throughput')
            axes[1, 0].set_xlabel('Steps')
            axes[1, 0].set_ylabel('Tokens/sec')
            axes[1, 0].grid(True, alpha=0.3)
        
        # System metrics plot
        system_metrics = self.get_system_metrics()
        if 'gpus' in system_metrics:
            gpu_loads = [gpu['gpu_load'] for gpu in system_metrics['gpus']]
            gpu_memory = [gpu['gpu_memory_percent'] for gpu in system_metrics['gpus']]
            gpu_ids = [f"GPU {gpu['gpu_id']}" for gpu in system_metrics['gpus']]
            
            x = range(len(gpu_ids))
            width = 0.35
            
            axes[1, 1].bar([i - width/2 for i in x], gpu_loads, width, label='GPU Load %', alpha=0.8)
            axes[1, 1].bar([i + width/2 for i in x], gpu_memory, width, label='GPU Memory %', alpha=0.8)
            axes[1, 1].set_title('Current GPU Status')
            axes[1, 1].set_xlabel('GPU')
            axes[1, 1].set_ylabel('Percentage')
            axes[1, 1].set_xticks(x)
            axes[1, 1].set_xticklabels(gpu_ids)
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.monitor_dir / f"training_progress_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Training progress plot saved to: {plot_path}")
        
        plt.show()
    
    def save_metrics(self):
        """Save current metrics to JSON file."""
        
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'system': self.get_system_metrics(),
            'training': self.parse_tensorboard_logs(),
            'checkpoints': self.get_checkpoint_info()
        }
        
        metrics_file = self.monitor_dir / f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        
        print(f"Metrics saved to: {metrics_file}")

def main():
    parser = argparse.ArgumentParser(description="Monitor Llama 3.1 Instruct 8B training")
    
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default='./checkpoints/llama31_instruct_8b',
        help='Checkpoint directory'
    )
    
    parser.add_argument(
        '--tensorboard-dir',
        type=str,
        default='./tensorboard_logs/llama31_instruct_8b',
        help='TensorBoard logs directory'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['report', 'plot', 'watch', 'metrics'],
        default='report',
        help='Monitoring mode'
    )
    
    parser.add_argument(
        '--interval',
        type=int,
        default=60,
        help='Update interval in seconds (for watch mode)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Output file for report'
    )
    
    args = parser.parse_args()
    
    monitor = TrainingMonitor(args.checkpoint_dir, args.tensorboard_dir)
    
    if args.mode == 'report':
        report = monitor.generate_report()
        if args.output:
            with open(args.output, 'w') as f:
                f.write(report)
            print(f"Report saved to: {args.output}")
        else:
            print(report)
    
    elif args.mode == 'plot':
        monitor.create_plots()
    
    elif args.mode == 'metrics':
        monitor.save_metrics()
    
    elif args.mode == 'watch':
        print(f"Monitoring training every {args.interval} seconds. Press Ctrl+C to stop.")
        try:
            while True:
                os.system('clear' if os.name == 'posix' else 'cls')
                report = monitor.generate_report()
                print(report)
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nMonitoring stopped.")

if __name__ == "__main__":
    main()
