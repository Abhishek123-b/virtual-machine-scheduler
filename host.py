from typing import List, Dict, Optional
import random
from .vm import VM
from src.constants.configs import VM_STATES, ENERGY_MODEL

class Host:
    """Enhanced host machine with advanced resource management."""
    
    def __init__(self, id: str, cpu_capacity: int = 100, memory_capacity: int = 100,
                 network_capacity: int = 1000, disk_capacity: int = 1000, gpu_capacity: int = 0):
        # Basic identification
        self.id = id
        
        # Resource capacities
        self.resources = {
            'cpu': {'capacity': cpu_capacity, 'used': 0},
            'memory': {'capacity': memory_capacity, 'used': 0},
            'network': {'capacity': network_capacity, 'used': 0},
            'disk': {'capacity': disk_capacity, 'used': 0},
            'gpu': {'capacity': gpu_capacity, 'used': 0}
        }
        
        # VM management
        self.vms: List[VM] = []
        
        # Resource monitoring
        self.utilization_history = {
            resource: {'usage': [], 'time': []} for resource in self.resources.keys()
        }
        
        # Health monitoring
        self.status = "Online"
        self.health_metrics = {
            'temperature': [],
            'power_consumption': [],
            'error_count': 0,
            'uptime': 0
        }
        
        # Failure handling
        self.failure_rate = 0.0001
        self.is_failed = False
        self.failure_history = []
        self.mtbf = 1000  # Mean Time Between Failures
        self.mttr = 10    # Mean Time To Repair
        
        # Performance tracking
        self.performance_metrics = {
            'response_time': [],
            'throughput': [],
            'availability': 100.0
        }
        
        # Energy management
        self.energy_profile = {
            'idle_power': 50,
            'max_power': 150,
            'power_efficiency': 0.8,
            'temperature_factor': 1.0
        }
        self.energy_consumption = 0
        self.power_history = []
        self.last_power_update = 0

    def reset_state(self) -> None:
        """Reset host to initial state."""
        self.resources = {resource: {'capacity': self.resources[resource]['capacity'], 'used': 0} for resource in self.resources}
        self.vms = []
        self.utilization_history = {resource: {'usage': [], 'time': []} for resource in self.resources}
        self.status = "Online"
        self.health_metrics = {
            'temperature': [],
            'power_consumption': [],
            'error_count': 0,
            'uptime': 0
        }
        self.failure_history = []
        self.energy_consumption = 0
        self.power_history = []
        self.last_power_update = 0

    def can_host_vm(self, vm: VM) -> bool:
        """Check if this host can accommodate the VM."""
        if self.is_failed:
            return False
            
        # Check all resource constraints
        for resource_type in self.resources:
            required = vm.resources[resource_type]['required']
            available = self.resources[resource_type]['capacity'] - self.resources[resource_type]['used']
            if required > available:
                return False
        return True

    def add_vm(self, vm: VM) -> bool:
        """Add a VM to this host with enhanced resource tracking."""
        if self.can_host_vm(vm):
            self.vms.append(vm)
            # Update resource usage
            for resource_type in self.resources:
                self.resources[resource_type]['used'] += vm.resources[resource_type]['required']
            
            vm.host = self
            vm.state = VM_STATES['RUNNING']
            vm.start_time = self.last_power_update
            vm.record_state(self.last_power_update, VM_STATES['RUNNING'])
            
            # Update performance metrics
            self._update_performance_metrics()
            return True
        return False

    def remove_vm(self, vm: VM) -> bool:
        """Remove a VM from this host with cleanup."""
        if vm in self.vms:
            self.vms.remove(vm)
            # Update resource usage
            for resource_type in self.resources:
                self.resources[resource_type]['used'] -= vm.resources[resource_type]['required']
            
            vm.completion_time = self.last_power_update
            vm.record_state(self.last_power_update, VM_STATES['COMPLETED'])
            vm.host = None
            
            # Update performance metrics
            self._update_performance_metrics()
            return True
        return False

    def _update_performance_metrics(self) -> None:
        """Update host performance metrics."""
        current_time = self.last_power_update
        
        # Calculate response time
        if self.vms:
            avg_response = sum(vm.execution_time for vm in self.vms) / len(self.vms)
            self.performance_metrics['response_time'].append((current_time, avg_response))
        
        # Calculate throughput
        completed_vms = sum(1 for vm in self.vms if vm.state == VM_STATES['COMPLETED'])
        if current_time > 0:
            throughput = completed_vms / current_time
            self.performance_metrics['throughput'].append((current_time, throughput))
        
        # Update availability
        uptime = current_time - sum(end - start for start, end in self.failure_history)
        self.performance_metrics['availability'] = (uptime / current_time * 100) if current_time > 0 else 100.0

    def update_resource_monitoring(self, current_time: float) -> None:
        """Update resource utilization history."""
        for resource_type in self.resources:
            usage_percent = (self.resources[resource_type]['used'] / 
                           self.resources[resource_type]['capacity'] * 100)
            self.utilization_history[resource_type]['usage'].append(usage_percent)
            self.utilization_history[resource_type]['time'].append(current_time)
        
        # Update health metrics
        self.health_metrics['temperature'].append(self._calculate_temperature())
        self._update_power_consumption(current_time)
        self.health_metrics['uptime'] = current_time

    def _calculate_temperature(self) -> float:
        """Calculate system temperature based on resource utilization."""
        cpu_util = self.resources['cpu']['used'] / self.resources['cpu']['capacity']
        base_temp = 35  # Base temperature in Celsius
        max_temp = 75   # Maximum temperature
        return min(base_temp + (max_temp - base_temp) * cpu_util, max_temp)

    def _update_power_consumption(self, current_time: float) -> None:
        """Update power consumption with temperature consideration."""
        if current_time - self.last_power_update >= 1:
            # Calculate base power
            cpu_util = self.resources['cpu']['used'] / self.resources['cpu']['capacity']
            base_power = (self.energy_profile['max_power'] - self.energy_profile['idle_power']) * cpu_util + self.energy_profile['idle_power']
            
            # Apply temperature factor
            temperature = self._calculate_temperature()
            temp_factor = 1.0 + max(0, (temperature - 50) / 100)  # Increase power usage above 50°C
            
            # Calculate final power
            power = base_power * temp_factor * self.energy_profile['power_efficiency']
            
            # Update metrics
            self.energy_consumption += power
            self.power_history.append((current_time, power))
            self.last_power_update = current_time

    def check_failure(self, current_time: float) -> bool:
        """Check if host fails or recovers."""
        if not self.is_failed:
            if random.random() < self.failure_rate:
                self.is_failed = True
                self.status = "Failed"
                return True
        else:
            if current_time % self.mttr == 0:
                self.is_failed = False
                self.status = "Online"
                return True
        return False

    def needs_maintenance(self, current_time: float) -> bool:
        """Check if host needs maintenance."""
        if current_time - self.last_power_update >= 1000:
            self.last_power_update = current_time
            return True
        return False

    def calculate_power_usage(self) -> float:
        """Calculate current power usage based on utilization."""
        cpu_util = self.resources['cpu']['used'] / self.resources['cpu']['capacity']
        power = (self.energy_profile['max_power'] - self.energy_profile['idle_power']) * cpu_util + self.energy_profile['idle_power']
        return power * self.energy_profile['power_efficiency']

    def update_power_history(self, current_time: float) -> None:
        """Update power consumption history."""
        if current_time - self.last_power_update >= 1:
            power = self.calculate_power_usage()
            self.energy_consumption += power
            self.power_history.append((current_time, power))
            self.last_power_update = current_time

    def get_utilization(self) -> tuple:
        """Get current CPU and memory utilization percentages."""
        cpu_util = (self.resources['cpu']['used'] / self.resources['cpu']['capacity']) * 100 if self.resources['cpu']['capacity'] > 0 else 0
        mem_util = (self.resources['memory']['used'] / self.resources['memory']['capacity']) * 100 if self.resources['memory']['capacity'] > 0 else 0
        return cpu_util, mem_util 