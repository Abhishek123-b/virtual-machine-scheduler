from typing import Dict, List
from src.models.host import Host
from src.models.vm import VM

class PerformanceMetrics:
    """Calculate and track performance metrics."""
    
    def __init__(self):
        self.metrics = {
            'throughput': 0.0,
            'average_latency': 0.0,
            'fairness_index': 0.0,
            'energy_efficiency': 0.0,
            'resource_utilization': {'cpu': 0.0, 'memory': 0.0},
            'vm_completion_times': [],
            'host_utilization_history': []
        }
        self.scheduler = None

    def calculate_metrics(self, scheduler) -> None:
        """Calculate all performance metrics."""
        if scheduler.current_time <= 0:
            return  # Don't calculate metrics if no time has passed
            
        self.scheduler = scheduler  # Store scheduler reference
        self._calculate_throughput(scheduler)
        self._calculate_latency(scheduler)
        self._calculate_fairness(scheduler)
        self._calculate_energy_efficiency(scheduler)
        self._calculate_resource_utilization(scheduler)

    def _calculate_throughput(self, scheduler) -> None:
        """Calculate system throughput."""
        if scheduler.current_time > 0:
            self.metrics['throughput'] = len(scheduler.completed_vms) / scheduler.current_time
        else:
            self.metrics['throughput'] = 0.0

    def _calculate_latency(self, scheduler) -> None:
        """Calculate average latency."""
        if scheduler.completed_vms:
            total_latency = 0
            for vm in scheduler.completed_vms:
                # Calculate actual execution time for completed VMs
                if vm.execution_time > 0:
                    total_latency += vm.execution_time
            self.metrics['average_latency'] = total_latency / len(scheduler.completed_vms)
        else:
            self.metrics['average_latency'] = 0.0

    def _calculate_fairness(self, scheduler) -> None:
        """Calculate Jain's fairness index."""
        completion_times = [vm.execution_time for vm in scheduler.completed_vms]
        if completion_times:
            mean_time = sum(completion_times) / len(completion_times)
            if mean_time > 0:
                squared_diffs = sum((t - mean_time) ** 2 for t in completion_times)
                self.metrics['fairness_index'] = 1 - (
                    squared_diffs / (len(completion_times) * mean_time ** 2)
                )
            else:
                self.metrics['fairness_index'] = 1.0  # Perfect fairness when all times are 0
        else:
            self.metrics['fairness_index'] = 1.0  # Perfect fairness when no VMs completed

    def _calculate_energy_efficiency(self, scheduler) -> None:
        """Calculate energy efficiency."""
        total_energy = sum(host.energy_consumption for host in scheduler.hosts)
        total_work = sum(vm.cpu_required * vm.memory_required 
                        for vm in scheduler.completed_vms)
        
        if total_energy > 0:
            self.metrics['energy_efficiency'] = total_work / total_energy
        else:
            self.metrics['energy_efficiency'] = 0.0

    def _calculate_resource_utilization(self, scheduler) -> None:
        """Calculate resource utilization."""
        if not scheduler.hosts:
            return
            
        # Calculate average utilization over time
        cpu_utils = []
        mem_utils = []
        
        for host in scheduler.hosts:
            if host.utilization_history['cpu'] and host.utilization_history['memory']:
                cpu_utils.extend(host.utilization_history['cpu'])
                mem_utils.extend(host.utilization_history['memory'])
        
        # Calculate average if we have data
        if cpu_utils and mem_utils:
            self.metrics['resource_utilization'] = {
                'cpu': sum(cpu_utils) / len(cpu_utils),
                'memory': sum(mem_utils) / len(mem_utils)
            }
        else:
            self.metrics['resource_utilization'] = {'cpu': 0.0, 'memory': 0.0}

    def generate_report(self) -> str:
        """Generate a formatted performance report."""
        if not self.scheduler:
            return "No simulation data available yet."
            
        # Calculate VM type distribution
        vm_types = {}
        for vm in self.scheduler.vms:
            vm_types[vm.vm_type] = vm_types.get(vm.vm_type, 0) + 1
        
        vm_type_str = "\n".join(f"- {vt}: {count} VMs" for vt, count in vm_types.items())
        
        return f"""
Performance Report
=================

Workload Distribution:
{vm_type_str}

Performance Metrics:
- Throughput: {self.metrics['throughput']:.2f} VMs/time unit
- Average Latency: {self.metrics['average_latency']:.2f} time units
- Fairness Index: {self.metrics['fairness_index']:.2f}
- Energy Efficiency: {self.metrics['energy_efficiency']:.2f} work/energy

Resource Utilization:
- CPU: {self.metrics['resource_utilization']['cpu']:.1f}%
- Memory: {self.metrics['resource_utilization']['memory']:.1f}%

Note: Fairness Index of 1.0 indicates perfect fairness in resource allocation
""" 