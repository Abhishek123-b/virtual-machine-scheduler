import simpy
import random
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import numpy as np
from matplotlib.figure import Figure
import threading
import time
from datetime import datetime
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors
from matplotlib import style
import tkinter.font as tkfont
import sys

# Set light style for matplotlib
style.use('default')

# Modern color palette with gradients and better aesthetics
COLORS = {
    'background': '#f8f9fa',  # Light gray background
    'secondary_bg': '#ffffff',  # Pure white
    'text': '#2c3e50',  # Dark blue-gray for text
    'accent': '#3498db',  # Bright blue
    'accent_dark': '#2980b9',  # Darker blue
    'grid': '#ecf0f1',  # Very light gray for grid
    'error': '#e74c3c',  # Bright red
    'success': '#2ecc71',  # Bright green
    'warning': '#f1c40f',  # Bright yellow
    'cpu_line': '#27ae60',  # Emerald green
    'mem_line': '#c0392b',  # Dark red
    'graph_bg': '#ffffff',  # White
    'button_bg': '#3498db',  # Blue
    'button_hover': '#2980b9',  # Darker blue
    'table_header': '#34495e',  # Dark blue-gray
    'table_row': '#ffffff',  # White
    'table_alt': '#f8f9fa',  # Light gray
    'border': '#bdc3c7',  # Light gray
    'highlight': '#3498db',  # Blue
    'text_secondary': '#7f8c8d',  # Gray
    'frame_bg': '#ffffff',  # White
    'button_text': '#ffffff',  # White
    'header_bg': '#34495e',  # Dark blue-gray
    'header_text': '#ffffff',  # White
    'primary': '#2196F3',      # Blue
    'secondary': '#FFC107',    # Amber
    'disabled': '#9E9E9E'      # Gray
}

# Add after COLORS dictionary
STYLE_CONFIG = {
    'Card.TLabelframe': {
        'background': COLORS['background'],
        'foreground': COLORS['text'],
        'borderwidth': 1,
        'relief': 'solid'
    },
    'TButton': {
        'background': COLORS['button_bg'],
        'foreground': COLORS['button_text'],
        'padding': (10, 5)
    },
    'TLabel': {
        'background': COLORS['background'],
        'foreground': COLORS['text']
    }
}

# Tooltips and help text
TOOLTIPS = {
    'host_name': 'Enter a unique name for the host',
    'host_cpu': 'Number of CPU cores available',
    'host_mem': 'Amount of memory in GB',
    'vm_name': 'Enter a unique name for the VM',
    'vm_cpu': 'Number of CPU cores required',
    'vm_mem': 'Amount of memory required in GB',
    'vm_priority': 'Priority level (1-10, lower is higher priority)',
    'vm_type': 'Type of workload for this VM',
    'algorithm': 'Choose the scheduling algorithm',
    'quantum': 'Time slice for Round Robin (in time units)',
    'speed': 'Simulation speed multiplier'
}

# Animation settings
ANIMATION_INTERVAL = 100  # Reduced from 1000 to 100 ms
DEFAULT_SIMULATION_SPEED = 5.0  # Increased default simulation speed

# ------------------------- VM States -------------------------
VM_STATES = {
    'PENDING': 'Pending',
    'RUNNING': 'Running',
    'COMPLETED': 'Completed',
    'MIGRATING': 'Migrating',
    'FAILED': 'Failed'
}

# ------------------------- VM Types -------------------------
VM_TYPES = {
    'Standard': {'cpu_weight': 0.5, 'mem_weight': 0.5},
    'High-CPU': {'cpu_weight': 0.7, 'mem_weight': 0.3},
    'High-Memory': {'cpu_weight': 0.3, 'mem_weight': 0.7},
    'Balanced': {'cpu_weight': 0.5, 'mem_weight': 0.5}
}

# Add after VM_TYPES dictionary
ENERGY_MODEL = {
    'idle_power': 50,  # Watts when idle
    'max_power': 200,  # Watts at full load
    'power_factor': 0.8  # Power efficiency factor
}

# Add after ENERGY_MODEL
SCENARIOS = {
    'steady_state': {
        'name': 'Steady State',
        'description': 'Constant workload with balanced resource requirements',
        'vm_count': 10,
        'cpu_range': (20, 40),
        'memory_range': (20, 40),
        'arrival_rate': 1.0
    },
    'traffic_spike': {
        'name': 'Traffic Spike',
        'description': 'Sudden increase in workload with varying resource demands',
        'vm_count': 20,
        'cpu_range': (30, 60),
        'memory_range': (30, 60),
        'arrival_rate': 0.5
    },
    'memory_intensive': {
        'name': 'Memory Intensive',
        'description': 'Workload with high memory requirements',
        'vm_count': 8,
        'cpu_range': (10, 30),
        'memory_range': (50, 80),
        'arrival_rate': 1.5
    },
    'cpu_intensive': {
        'name': 'CPU Intensive',
        'description': 'Workload with high CPU requirements',
        'vm_count': 8,
        'cpu_range': (50, 80),
        'memory_range': (10, 30),
        'arrival_rate': 1.5
    }
}

# Add after SCENARIOS dictionary
TUTORIALS = {
    'getting_started': {
        'title': 'Getting Started',
        'steps': [
            "1. Select a scheduling algorithm from the dropdown menu",
            "2. Choose a test scenario or create custom VMs",
            "3. Click 'Start Simulation' to begin",
            "4. Monitor the graphs and sequence table",
            "5. View the performance report when complete"
        ]
    },
    'algorithms': {
        'title': 'Understanding Algorithms',
        'steps': [
            "First Fit: Assigns VMs to the first available host",
            "Round Robin: Distributes VMs equally across hosts",
            "Priority: Schedules based on VM priority levels",
            "CFS: Ensures fair CPU time distribution"
        ]
    },
    'metrics': {
        'title': 'Performance Metrics',
        'steps': [
            "Throughput: VMs completed per time unit",
            "Latency: Time from VM creation to completion",
            "Fairness: How evenly resources are distributed",
            "Energy Efficiency: Work done per energy unit"
        ]
    }
}

# Add after SCHEDULING_STEPS dictionary
SCHEDULING_STEPS = {
    'fcfs': [
        ("Arrival", "VMs are ordered by arrival time"),
        ("Resource Check", "Check if host has enough CPU and memory"),
        ("Assignment", "VM is assigned to first available host"),
        ("Execution", "VM runs until completion without interruption")
    ],
    'round_robin': [
        ("Queue Setup", "VMs are placed in a circular queue"),
        ("Time Slice", "Each VM gets a fixed quantum of time"),
        ("Context Switch", "After quantum expires, VM is preempted"),
        ("Rotation", "Preempted VM goes to back of queue"),
        ("Continuation", "Next VM in queue gets CPU time")
    ],
    'priority': [
        ("Priority Sort", "VMs are sorted by priority level"),
        ("Resource Check", "Check host resource availability"),
        ("Assignment", "Highest priority VM gets resources first"),
        ("Preemption", "Lower priority VMs may be preempted"),
        ("Dynamic Update", "Priorities may be adjusted to prevent starvation")
    ],
    'cfs': [
        ("Load Calculation", "Calculate virtual runtime for each VM"),
        ("Fairness Check", "Compare virtual runtime across VMs"),
        ("Selection", "Select VM with lowest virtual runtime"),
        ("Execution", "Run selected VM for a time slice"),
        ("Update", "Update virtual runtime and rebalance")
    ]
}

# ------------------------- Algorithm Information -------------------------
ALGORITHM_INFO = {
    'First Fit': {
        'name': 'First Fit Algorithm',
        'description': 'Assigns each process to the first available host that can accommodate its resource requirements.',
        'pros': '• Simple and fast execution\n• Low computational overhead\n• Good for real-time scheduling',
        'cons': '• May lead to fragmentation\n• Not optimal for resource utilization\n• Performance degrades over time',
        'tips': '• Best for scenarios with uniform resource requirements\n• Regular defragmentation recommended\n• Monitor resource utilization patterns'
    },
    'Best Fit': {
        'name': 'Best Fit Algorithm',
        'description': 'Places each process in the host with the smallest sufficient space, minimizing wasted resources.',
        'pros': '• Efficient resource utilization\n• Reduces internal fragmentation\n• Better long-term performance',
        'cons': '• Higher computational overhead\n• May be slower than First Fit\n• Can lead to external fragmentation',
        'tips': '• Ideal for varied resource requirements\n• Consider resource prediction\n• Regular optimization checks'
    },
    'Worst Fit': {
        'name': 'Worst Fit Algorithm',
        'description': 'Assigns processes to hosts with the largest available space, maximizing remaining contiguous space.',
        'pros': '• Maintains large free blocks\n• Good for varying process sizes\n• Reduces external fragmentation',
        'cons': '• Poor space utilization\n• Not efficient for uniform workloads\n• Higher memory overhead',
        'tips': '• Use when process sizes vary greatly\n• Monitor large block availability\n• Consider hybrid approaches'
    },
    'Round Robin': {
        'name': 'Round Robin Algorithm',
        'description': 'Distributes processes sequentially across available hosts in a circular manner.',
        'pros': '• Fair distribution\n• Simple implementation\n• Prevents host overload',
        'cons': '• Not resource-aware\n• May underutilize resources\n• Not suitable for heterogeneous environments',
        'tips': '• Best for homogeneous environments\n• Set appropriate time quantum\n• Monitor load distribution'
    }
}

# ------------------------- Host Class -------------------------
class Host:
    def __init__(self, id, cpu_capacity=100, memory_capacity=100):
        self.id = id
        self.cpu_capacity = cpu_capacity
        self.memory_capacity = memory_capacity
        self.cpu_used = 0
        self.memory_used = 0
        self.vms = []
        self.utilization_history = {'cpu': [], 'memory': []}
        self.status = "Online"  # Online, Offline, Maintenance
        self.failure_rate = 0.001  # Probability of host failure per time unit
        self.is_failed = False
        self.recovery_time = 10  # Time units needed for recovery
        self.last_maintenance = 0
        self.maintenance_interval = 100  # Time units between maintenance
        self.energy_consumption = 0
        self.power_history = []
        self.last_power_update = 0

    def reset_state(self):
        """Reset host to initial state."""
        self.cpu_used = 0
        self.memory_used = 0
        self.vms = []
        self.utilization_history = {'cpu': [], 'memory': []}
        self.status = "Online"

    def can_host_vm(self, vm):
        """Check if this host can accommodate the VM"""
        return (not self.is_failed and 
                self.cpu_used + vm.cpu <= self.cpu_capacity and 
                self.memory_used + vm.memory <= self.memory_capacity)

    def add_vm(self, vm):
        """Add a VM to this host"""
        if self.can_host_vm(vm):
            self.vms.append(vm)
            self.cpu_used += vm.cpu
            self.memory_used += vm.memory
            vm.host = self
            vm.state = VM_STATES['RUNNING']
            return True
        return False

    def remove_vm(self, vm):
        """Remove a VM from this host"""
        if vm in self.vms:
            self.vms.remove(vm)
            self.cpu_used -= vm.cpu
            self.memory_used -= vm.memory
            vm.host = None
            vm.state = VM_STATES['PENDING']
            return True
        return False

    def get_utilization(self):
        """Calculate current CPU and memory utilization"""
        cpu_util = (self.cpu_capacity - self.cpu_used) / self.cpu_capacity * 100
        mem_util = (self.memory_capacity - self.memory_used) / self.memory_capacity * 100
        return cpu_util, mem_util

    def check_failure(self, current_time):
        """Check if host fails and handle recovery"""
        if not self.is_failed and random.random() < self.failure_rate:
            self.is_failed = True
            # Migrate all VMs to other hosts
            for vm in self.vms[:]:
                self.remove_vm(vm)
            return True
        elif self.is_failed and current_time - self.last_maintenance >= self.recovery_time:
            self.is_failed = False
            self.last_maintenance = current_time
            return True
        return False

    def needs_maintenance(self, current_time):
        """Check if host needs maintenance"""
        return current_time - self.last_maintenance >= self.maintenance_interval

    def calculate_power_usage(self):
        """Calculate current power usage based on utilization"""
        cpu_util = self.cpu_used / self.cpu_capacity
        power = (ENERGY_MODEL['max_power'] - ENERGY_MODEL['idle_power']) * cpu_util + ENERGY_MODEL['idle_power']
        return power * ENERGY_MODEL['power_factor']

    def update_power_history(self, current_time):
        """Update power consumption history"""
        if current_time - self.last_power_update >= 1:  # Update every time unit
            power = self.calculate_power_usage()
            self.energy_consumption += power
            self.power_history.append((current_time, power))
            self.last_power_update = current_time

# ------------------------- VM Class -------------------------
class VM:
    def __init__(self, id, cpu_required, memory_required, vm_type="Standard"):
        self.id = id
        self.cpu_required = cpu_required
        self.memory_required = memory_required
        self.vm_type = vm_type
        self.state = "PENDING"
        self.host = None
        self.waiting_time = 0
        self.execution_time = 0
        self.priority = 1
        self.state_history = []
        self.migration_count = 0
        self.required_time = self._calculate_required_time()
        self.remaining_time = self.required_time
        self.cpu_usage_history = []
        self.mem_usage_history = []

    def _calculate_required_time(self):
        """Calculate required execution time based on VM type and resources"""
        base_time = 100  # Base time units
        # Default to Standard if vm_type not found
        weights = VM_TYPES.get(self.vm_type, VM_TYPES['Standard'])
        weighted_resources = (self.cpu_required * weights['cpu_weight'] + 
                            self.memory_required * weights['mem_weight'])
        return max(base_time / (weighted_resources + 1), 10)  # Minimum time of 10 units

    def record_state(self, time, state):
        """Record VM state change"""
        self.state = state
        self.state_history.append((time, state))
        
        if state == VM_STATES['RUNNING'] and self.execution_time == 0:
            self.execution_time = time
        elif state == VM_STATES['COMPLETED']:
            self.execution_time = time

    def update_usage(self, time, cpu_usage, mem_usage):
        """Update resource usage history"""
        self.cpu_usage_history.append((time, cpu_usage))
        self.mem_usage_history.append((time, mem_usage))

    def migrate(self, new_host):
        """Migrate VM to a new host"""
        if self.host:
            self.host.remove_vm(self)
        if new_host.add_vm(self):
            self.migration_count += 1
            self.state = VM_STATES['MIGRATING']
            return True
        return False

    def is_completed(self):
        """Check if VM has completed its execution"""
        return self.state == VM_STATES['COMPLETED']

    def update_progress(self, time_step):
        """Update VM progress"""
        if self.state == VM_STATES['RUNNING']:
            self.remaining_time -= time_step
            if self.remaining_time <= 0:
                self.state = VM_STATES['COMPLETED']
                self.execution_time = time_step
                if self.host:
                    self.host.remove_vm(self)

    def reset_state(self):
        """Reset VM to initial state."""
        self.state = "PENDING"
        self.host = None
        self.waiting_time = 0
        self.execution_time = 0
        self.state_history = []
        self.migration_count = 0
        self.remaining_time = self.required_time
        self.cpu_usage_history = []
        self.mem_usage_history = []

# ------------------------- VMScheduler Class -------------------------
class VMScheduler:
    def __init__(self, env, hosts, vms, algorithm, quantum):
        self.env = env
        self.hosts = hosts
        self.vms = vms
        self.algorithm = algorithm
        self.quantum = quantum
        self.completed_vms = []
        self.current_time = 0
        self.migration_threshold = 0.8  # CPU/Memory utilization threshold for migration
        self.load_balancing_interval = 10  # Time units between load balancing checks
        self.virtual_runtime = {}  # For CFS
        self.min_granularity = 1  # Minimum time slice for CFS
        self.target_latency = 6  # Target latency for CFS

    def run(self):
        step_num = 0
        algorithm = self.algorithm.lower()
        
        # Log initial state
        self.log_step(step_num, "Initialization", 
                     f"Starting {algorithm.upper()} scheduling", 
                     self.env.now, "Started")
        step_num += 1
        
        # Log algorithm-specific steps
        for action, description in SCHEDULING_STEPS[algorithm]:
            self.log_step(step_num, action, description, 
                         self.env.now, "In Progress")
            step_num += 1
            
            yield self.env.timeout(1)
        
        while not self.is_completed():
            # Check for host failures and maintenance
            self._check_hosts()
            
            # Schedule VMs according to the selected algorithm
            self.schedule_vms()
            
            # Update VM progress
            self._update_vm_progress()
            
            # Perform load balancing periodically
            if self.current_time % self.load_balancing_interval == 0:
                self._balance_load()
                self.log_step(step_num, "Load Balancing", 
                            "Redistributing VMs for optimal resource usage",
                            self.env.now, "Completed")
                step_num += 1
            
            yield self.env.timeout(1)
            self.current_time += 1

    def _check_hosts(self):
        """Check host status and handle failures/maintenance"""
        for host in self.hosts:
            if host.check_failure(self.current_time):
                print(f"Host {host.id} {'failed' if host.is_failed else 'recovered'}")
            
            if host.needs_maintenance(self.current_time):
                print(f"Host {host.id} needs maintenance")
                # Implement maintenance logic here

    def _update_vm_progress(self):
        """Update progress of all running VMs"""
        for vm in self.vms:
            if vm.state == VM_STATES['RUNNING']:
                vm.update_progress(1)
                if vm.is_completed():
                    self.completed_vms.append(vm)

    def _balance_load(self):
        """Balance load across hosts"""
        for host in self.hosts:
            if host.is_failed:
                continue
                
            cpu_util, mem_util = host.get_utilization()
            if cpu_util > self.migration_threshold * 100 or mem_util > self.migration_threshold * 100:
                # Find a less loaded host
                for target_host in self.hosts:
                    if (target_host != host and not target_host.is_failed and
                        target_host.get_utilization()[0] < self.migration_threshold * 100 and
                        target_host.get_utilization()[1] < self.migration_threshold * 100):
                        
                        # Try to migrate a VM
                        for vm in host.vms[:]:
                            if target_host.can_host_vm(vm):
                                vm.migrate(target_host)
                                break

    def schedule_vms(self):
        if self.algorithm == 'fcfs':
            self._schedule_fcfs()
        elif self.algorithm == 'round_robin':
            self._schedule_round_robin()
        elif self.algorithm == 'priority':
            self._schedule_priority()
        elif self.algorithm == 'cfs':
            self._schedule_cfs()

    def _schedule_fcfs(self):
        pending_vms = [vm for vm in self.vms if vm not in self.completed_vms and vm.state == VM_STATES['PENDING']]
        
        for vm in pending_vms:
            for host in self.hosts:
                if host.can_host_vm(vm):
                    host.add_vm(vm)
                    break

    def _schedule_round_robin(self):
        for host in self.hosts:
            if host.vms and not host.is_failed:
                current_vm = host.vms.pop(0)
                host.remove_vm(current_vm)
                host.add_vm(current_vm)

    def _schedule_priority(self):
        pending_vms = sorted([vm for vm in self.vms if vm not in self.completed_vms and vm.state == VM_STATES['PENDING']],
                           key=lambda x: x.priority)
        
        for vm in pending_vms:
            for host in self.hosts:
                if host.can_host_vm(vm):
                    host.add_vm(vm)
                    break

    def _schedule_cfs(self):
        """Completely Fair Scheduler implementation"""
        if not self.virtual_runtime:
            # Initialize virtual runtime for all VMs
            for vm in self.vms:
                self.virtual_runtime[vm.id] = 0

        # Calculate target latency and minimum granularity
        target_latency = self.target_latency
        min_granularity = self.min_granularity

        # Find VM with minimum virtual runtime
        min_vm = None
        min_vruntime = float('inf')
        
        for vm in self.vms:
            if vm.state == VM_STATES['PENDING']:
                if self.virtual_runtime[vm.id] < min_vruntime:
                    min_vruntime = self.virtual_runtime[vm.id]
                    min_vm = vm

        if min_vm:
            # Calculate time slice
            time_slice = min(target_latency, min_granularity)
            
            # Find suitable host
            for host in self.hosts:
                if host.can_host_vm(min_vm):
                    host.add_vm(min_vm)
                    # Update virtual runtime
                    self.virtual_runtime[min_vm.id] += time_slice
                    break

    def is_completed(self):
        return len(self.completed_vms) == len(self.vms)

    def log_step(self, step_num, action, description, time, status):
        # This will be called by the GUI to update the sequence table
        if hasattr(self, 'gui'):
            self.gui.sequence_table.update_sequence(
                step_num, action, description, f"t={time:.1f}", status)

# ------------------------- GUI Setup -------------------------
class ToolTip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip = None
        self.widget.bind('<Enter>', self.show_tooltip)
        self.widget.bind('<Leave>', self.hide_tooltip)

    def show_tooltip(self, event=None):
        x, y, _, _ = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 25

        self.tooltip = tk.Toplevel(self.widget)
        self.tooltip.wm_overrideredirect(True)
        self.tooltip.wm_geometry(f"+{x}+{y}")

        label = ttk.Label(self.tooltip, text=self.text, justify='left',
                         background=COLORS['background'], foreground=COLORS['text'],
                         relief='solid', borderwidth=1, padding=(5, 5))
        label.pack()

    def hide_tooltip(self, event=None):
        if self.tooltip:
            self.tooltip.destroy()
            self.tooltip = None

class InteractiveGraph(ttk.Frame):
    def __init__(self, parent, title, ylabel):
        super().__init__(parent)
        self.fig = Figure(figsize=(6, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self)
        self.toolbar.update()
        
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.ax.set_title(title, color=COLORS['text'])
        self.ax.set_ylabel(ylabel, color=COLORS['text'])
        self.ax.set_xlabel('Time', color=COLORS['text'])
        self.configure_style()

    def configure_style(self):
        self.fig.patch.set_facecolor(COLORS['background'])
        self.ax.set_facecolor(COLORS['background'])
        self.ax.tick_params(colors=COLORS['text'])
        for spine in self.ax.spines.values():
            spine.set_color(COLORS['text'])
        self.ax.grid(True, color=COLORS['grid'], linestyle='--', alpha=0.3)

class SequenceTable(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.create_table()

    def create_table(self):
        # Create Treeview
        columns = ('Step', 'Action', 'Description', 'Time', 'Status')
        self.tree = ttk.Treeview(self, columns=columns, show='headings', height=8)
        
        # Configure columns
        self.tree.heading('Step', text='Step')
        self.tree.heading('Action', text='Action')
        self.tree.heading('Description', text='Description')
        self.tree.heading('Time', text='Time')
        self.tree.heading('Status', text='Status')
        
        self.tree.column('Step', width=50)
        self.tree.column('Action', width=100)
        self.tree.column('Description', width=300)
        self.tree.column('Time', width=100)
        self.tree.column('Status', width=100)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(self, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        # Pack elements
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def update_sequence(self, step_num, action, description, time, status):
        """Update the sequence table with new information."""
        item_id = f"step_{step_num}"
        
        # Check if step already exists
        existing_items = self.tree.get_children()
        if item_id in existing_items:
            self.tree.item(item_id, values=(step_num, action, description, time, status))
        else:
            self.tree.insert('', 'end', iid=item_id, 
                           values=(step_num, action, description, time, status))
        
        # Ensure the latest entry is visible
        self.tree.see(item_id)

    def clear(self):
        """Clear all entries in the sequence table."""
        for item in self.tree.get_children():
            self.tree.delete(item)

class PerformanceMetrics:
    def __init__(self):
        self.metrics = {
            'throughput': 0,
            'average_latency': 0,
            'fairness_index': 0,
            'energy_efficiency': 0,
            'resource_utilization': {'cpu': 0, 'memory': 0},
            'vm_completion_times': [],
            'host_utilization_history': []
        }

    def calculate_metrics(self, scheduler):
        """Calculate performance metrics"""
        # Calculate throughput
        self.metrics['throughput'] = len(scheduler.completed_vms) / scheduler.current_time

        # Calculate average latency
        if scheduler.completed_vms:
            total_latency = sum(vm.execution_time - vm.waiting_time for vm in scheduler.completed_vms)
            self.metrics['average_latency'] = total_latency / len(scheduler.completed_vms)

        # Calculate fairness index
        completion_times = [vm.execution_time for vm in scheduler.completed_vms]
        if completion_times:
            mean_time = sum(completion_times) / len(completion_times)
            squared_diffs = sum((t - mean_time) ** 2 for t in completion_times)
            self.metrics['fairness_index'] = 1 - (squared_diffs / (len(completion_times) * mean_time ** 2))

        # Calculate energy efficiency
        total_energy = sum(host.energy_consumption for host in scheduler.hosts)
        total_work = sum(vm.cpu_required * vm.memory_required for vm in scheduler.completed_vms)
        self.metrics['energy_efficiency'] = total_work / total_energy if total_energy > 0 else 0

        # Calculate resource utilization
        cpu_util = sum(host.cpu_used for host in scheduler.hosts) / sum(host.cpu_capacity for host in scheduler.hosts)
        mem_util = sum(host.memory_used for host in scheduler.hosts) / sum(host.memory_capacity for host in scheduler.hosts)
        self.metrics['resource_utilization'] = {'cpu': cpu_util, 'memory': mem_util}

    def generate_report(self):
        """Generate a performance report"""
        report = f"""
Performance Report
=================

Throughput: {self.metrics['throughput']:.2f} VMs/time unit
Average Latency: {self.metrics['average_latency']:.2f} time units
Fairness Index: {self.metrics['fairness_index']:.2f}
Energy Efficiency: {self.metrics['energy_efficiency']:.2f} work/energy

Resource Utilization:
- CPU: {self.metrics['resource_utilization']['cpu']*100:.1f}%
- Memory: {self.metrics['resource_utilization']['memory']*100:.1f}%
"""
        return report

class VMSchedulerGUI(ttk.Frame):
    def __init__(self, root):
        super().__init__(root)
        self.root = root
        self.root.title("VM Scheduler Simulator")
        
        # Get screen dimensions
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        
        # Calculate window size (80% of screen)
        window_width = int(screen_width * 0.8)
        window_height = int(screen_height * 0.8)
        
        # Calculate position
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        
        # Set window size and position
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")
        self.root.minsize(800, 600)  # Set minimum window size
        
        # Make window resizable
        self.root.resizable(True, True)
        
        # Initialize variables
        self.simulation_running = False
        self.simulation_paused = False
        self.simulation_thread = None
        self.hosts = []
        self.vms = []
        self.env = None
        self.scheduler = None
        
        # Initialize GUI variables
        self.quantum_var = tk.StringVar(value="2.0")
        self.speed_var = tk.StringVar(value=str(DEFAULT_SIMULATION_SPEED))
        self.host_cpu_var = tk.StringVar()
        self.host_memory_var = tk.StringVar()
        self.vm_cpu_var = tk.StringVar()
        self.vm_memory_var = tk.StringVar()
        self.vm_priority_var = tk.StringVar()
        self.vm_type_var = tk.StringVar(value="Standard")
        self.current_algorithm = tk.StringVar(value="First Fit")
        self.performance_metrics = PerformanceMetrics()
        self.scenario_var = tk.StringVar(value='steady_state')
        self.tutorial_var = tk.StringVar(value='getting_started')
        
        # Configure root window
        self.root.configure(bg=COLORS['background'])
        
        # Apply styles
        self.style = ttk.Style()
        self.style.configure('Main.TFrame', background=COLORS['background'])
        for widget, config in STYLE_CONFIG.items():
            self.style.configure(widget, **config)
        
        # Create main container with padding
        self.main_container = ttk.Frame(self.root, padding="10", style='Main.TFrame')
        self.main_container.grid(row=0, column=0, sticky="nsew")
        
        # Configure grid weights for responsiveness
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        self.main_container.grid_rowconfigure(0, weight=1)
        self.main_container.grid_columnconfigure(1, weight=3)  # Give more weight to visualization
        
        # Create left panel (controls and info)
        self.left_panel = ttk.Frame(self.main_container, padding="5", style='Main.TFrame')
        self.left_panel.grid(row=0, column=0, sticky="nsew", padx=5)
        self.left_panel.grid_columnconfigure(0, weight=1)
        
        # Create right panel (visualization and tables)
        self.right_panel = ttk.Frame(self.main_container, padding="5", style='Main.TFrame')
        self.right_panel.grid(row=0, column=1, sticky="nsew", padx=5)
        self.right_panel.grid_columnconfigure(0, weight=1)
        
        # Initialize components with proper weights
        self.create_control_panel()
        self.create_vm_table()
        self.create_sequence_frame()
        self.create_info_frame()
        self.create_visualization_frame()
        
        # Initialize algorithm information
        self.root.after(100, lambda: self._update_algorithm_display())
        
        # Initialize at least one default host
        default_host = Host("Host-1", cpu_capacity=100, memory_capacity=100)
        self.hosts = [default_host]

    def _update_algorithm_display(self):
        """Update the algorithm information display."""
        try:
            algorithm = self.current_algorithm.get()
            if algorithm in ALGORITHM_INFO:
                info = ALGORITHM_INFO[algorithm]
                self.info_text.configure(state='normal')
                self.info_text.delete('1.0', tk.END)
                
                # Add title
                self.info_text.tag_configure('title', font=('Segoe UI', 12, 'bold'))
                self.info_text.insert(tk.END, f"{info['name']}\n\n", 'title')
                
                # Add description
                self.info_text.tag_configure('heading', font=('Segoe UI', 10, 'bold'))
                self.info_text.insert(tk.END, "Description:\n", 'heading')
                self.info_text.insert(tk.END, f"{info['description']}\n\n")
                
                # Add pros
                self.info_text.insert(tk.END, "Advantages:\n", 'heading')
                self.info_text.insert(tk.END, f"{info['pros']}\n\n")
                
                # Add cons
                self.info_text.insert(tk.END, "Disadvantages:\n", 'heading')
                self.info_text.insert(tk.END, f"{info['cons']}\n\n")
                
                # Add tips
                self.info_text.insert(tk.END, "Best Practices:\n", 'heading')
                self.info_text.insert(tk.END, f"{info['tips']}")
                
                # Make text read-only
                self.info_text.configure(state='disabled')
        except Exception as e:
            print(f"Error updating algorithm info: {e}")

    def _on_algorithm_change(self, *args):
        """Handle algorithm change events."""
        self.root.after(100, lambda: self._update_algorithm_display())

    def create_control_panel(self):
        control_frame = ttk.LabelFrame(self.left_panel, text="Control Panel", padding="10", style='Card.TLabelframe')
        control_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        control_frame.grid_columnconfigure(1, weight=1)  # Make second column expandable
        
        # Algorithm selection with more space
        algorithm_label = ttk.Label(control_frame, text="Select Algorithm:", font=('Segoe UI', 10))
        algorithm_label.grid(row=0, column=0, sticky="w", pady=10, padx=5)
        
        algorithms = list(ALGORITHM_INFO.keys())
        algorithm_combo = ttk.Combobox(control_frame, textvariable=self.current_algorithm, 
                                     values=algorithms, state="readonly", width=30)
        algorithm_combo.grid(row=0, column=1, sticky="ew", padx=5, pady=10, columnspan=2)
         
        # Bind algorithm selection events
        algorithm_combo.bind('<<ComboboxSelected>>', self._on_algorithm_change)
        self.current_algorithm.trace('w', self._on_algorithm_change)
        
        # VM controls with better spacing
        vm_frame = ttk.Frame(control_frame)
        vm_frame.grid(row=1, column=0, columnspan=3, sticky="ew", pady=10)
        vm_frame.grid_columnconfigure(1, weight=1)
        vm_frame.grid_columnconfigure(3, weight=1)
        
        # CPU input
        ttk.Label(vm_frame, text="CPU Cores:", font=('Segoe UI', 10)).grid(row=0, column=0, sticky="w", padx=5)
        cpu_entry = ttk.Entry(vm_frame, textvariable=self.vm_cpu_var, width=10)
        cpu_entry.grid(row=0, column=1, sticky="w", padx=5)
        
        # Memory input
        ttk.Label(vm_frame, text="Memory (GB):", font=('Segoe UI', 10)).grid(row=0, column=2, sticky="w", padx=5)
        mem_entry = ttk.Entry(vm_frame, textvariable=self.vm_memory_var, width=10)
        mem_entry.grid(row=0, column=3, sticky="w", padx=5)
        
        # Add VM button
        add_button = ttk.Button(vm_frame, text="Add VM", command=self.add_vm, width=15)
        add_button.grid(row=0, column=4, padx=10)
        
        # Control buttons with better layout
        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=2, column=0, columnspan=3, sticky="ew", pady=15)
        
        self.start_button = ttk.Button(button_frame, text="Start Simulation", 
                                     command=self.start_simulation, width=15)
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        self.pause_button = ttk.Button(button_frame, text="Pause", 
                                     command=self.pause_simulation, width=10,
                                     state=tk.DISABLED)
        self.pause_button.pack(side=tk.LEFT, padx=5)
        
        self.reset_button = ttk.Button(button_frame, text="Reset", 
                                     command=self.reset_simulation, width=10)
        self.reset_button.pack(side=tk.LEFT, padx=5)
        
        # Speed control with label
        speed_frame = ttk.Frame(control_frame)
        speed_frame.grid(row=3, column=0, columnspan=3, sticky="ew", pady=5)
        
        speed_label = ttk.Label(speed_frame, text="Simulation Speed:", font=('Segoe UI', 10))
        speed_label.pack(side=tk.LEFT, padx=5)
        
        self.speed_scale = ttk.Scale(speed_frame, from_=0.5, to=2.0, 
                                   orient=tk.HORIZONTAL, length=200)
        self.speed_scale.set(1.0)
        self.speed_scale.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)
        
        # Scenario selection with better layout
        scenario_frame = ttk.LabelFrame(control_frame, text="Test Scenarios", padding="10")
        scenario_frame.grid(row=4, column=0, columnspan=3, sticky="ew", pady=10)
        scenario_frame.grid_columnconfigure(0, weight=1)
        
        scenario_combo = ttk.Combobox(scenario_frame, textvariable=self.scenario_var,
                                    values=list(SCENARIOS.keys()), state="readonly", width=40)
        scenario_combo.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        
        scenario_desc = ttk.Label(scenario_frame, text="", wraplength=400, 
                                font=('Segoe UI', 9), justify=tk.LEFT)
        scenario_desc.grid(row=1, column=0, sticky="w", padx=5, pady=5)
        
        def update_scenario_desc(*args):
            scenario = SCENARIOS[self.scenario_var.get()]
            scenario_desc.configure(text=scenario['description'])
        
        self.scenario_var.trace('w', update_scenario_desc)
        update_scenario_desc()

    def create_vm_table(self):
        """Create VM status table."""
        vm_table_frame = ttk.LabelFrame(self.right_panel, text="Virtual Machines", 
                                      style='Card.TLabelframe', padding=15)
        vm_table_frame.grid(row=1, column=0, sticky="nsew", padx=15, pady=15)
        vm_table_frame.grid_columnconfigure(0, weight=1)
        vm_table_frame.grid_rowconfigure(0, weight=1)
        
        # Create table
        columns = ('ID', 'CPU', 'Memory', 'Type', 'Priority', 'Host', 'Status', 'Time')
        self.vm_table = ttk.Treeview(vm_table_frame, columns=columns,
                                   show="headings", height=6)
        
        # Configure columns
        widths = {
            'ID': 80, 'CPU': 60, 'Memory': 80, 'Type': 100,
            'Priority': 60, 'Host': 80, 'Status': 100, 'Time': 80
        }
        for col in columns:
            self.vm_table.heading(col, text=col)
            self.vm_table.column(col, width=widths[col])
        
        # Add scrollbars
        y_scrollbar = ttk.Scrollbar(vm_table_frame, orient="vertical",
                                  command=self.vm_table.yview)
        x_scrollbar = ttk.Scrollbar(vm_table_frame, orient="horizontal",
                                  command=self.vm_table.xview)
        
        self.vm_table.configure(yscrollcommand=y_scrollbar.set,
                              xscrollcommand=x_scrollbar.set)
        
        # Grid layout
        self.vm_table.grid(row=0, column=0, sticky="nsew")
        y_scrollbar.grid(row=0, column=1, sticky="ns")
        x_scrollbar.grid(row=1, column=0, sticky="ew")

    def create_sequence_frame(self):
        """Create sequence frame with improved visibility."""
        self.sequence_frame = ttk.LabelFrame(self.left_panel, text="Execution Sequence",
                                           style='Card.TLabelframe', padding=15)
        self.sequence_frame.grid(row=1, column=0, sticky="nsew", padx=15, pady=15)
        self.sequence_frame.grid_columnconfigure(0, weight=1)
        self.sequence_frame.grid_rowconfigure(0, weight=1)
        
        # Create the sequence table
        self.sequence_table = SequenceTable(self.sequence_frame)
        self.sequence_table.grid(row=0, column=0, sticky="nsew")

    def create_info_frame(self):
        """Create info frame with improved visibility."""
        self.info_frame = ttk.LabelFrame(self.left_panel, text="Algorithm Information",
                                       style='Card.TLabelframe', padding=15)
        self.info_frame.grid(row=6, column=0, columnspan=2, sticky="nsew", 
                            padx=15, pady=15)
        
        # Create text widget with improved styling
        self.info_text = tk.Text(self.info_frame, wrap=tk.WORD, 
                                bg=COLORS['background'],
                                fg=COLORS['text'],
                                font=('Segoe UI', 10),
                                height=6)  # Show 6 lines at a time
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(self.info_frame, orient=tk.VERTICAL,
                                command=self.info_text.yview)
        self.info_text.configure(yscrollcommand=scrollbar.set)
        
        # Pack widgets
        self.info_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Add tutorial section
        tutorial_frame = ttk.LabelFrame(self.left_panel, text="Tutorials", 
                                      style='Card.TLabelframe', padding=15)
        tutorial_frame.grid(row=7, column=0, columnspan=2, sticky="nsew", 
                          padx=15, pady=15)
        
        # Tutorial selection
        tutorial_combo = ttk.Combobox(tutorial_frame, textvariable=self.tutorial_var,
                                    values=list(TUTORIALS.keys()), state="readonly")
        tutorial_combo.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        
        # Tutorial content
        tutorial_text = tk.Text(tutorial_frame, wrap=tk.WORD, height=6,
                              bg=COLORS['background'], fg=COLORS['text'])
        tutorial_text.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        
        def update_tutorial(*args):
            tutorial = TUTORIALS[self.tutorial_var.get()]
            tutorial_text.delete('1.0', tk.END)
            tutorial_text.insert(tk.END, f"{tutorial['title']}\n\n")
            for step in tutorial['steps']:
                tutorial_text.insert(tk.END, f"• {step}\n")
            tutorial_text.configure(state='disabled')
        
        self.tutorial_var.trace('w', update_tutorial)
        update_tutorial()

    def create_visualization_frame(self):
        """Create and style the visualization frame with modern light theme."""
        self.visualization_frame = ttk.LabelFrame(self.right_panel, text="Resource Utilization", 
                                                style='Card.TLabelframe', padding=15)
        self.visualization_frame.grid(row=0, column=0, sticky="nsew", padx=15, pady=15)
        self.visualization_frame.grid_rowconfigure(0, weight=1)
        self.visualization_frame.grid_columnconfigure(0, weight=1)
        
        # Create figure with light theme and responsive size
        plt.style.use('default')
        self.fig = Figure(dpi=100, facecolor=COLORS['background'])
        
        # Create subplots with GridSpec for better control
        gs = self.fig.add_gridspec(2, 2, height_ratios=[2, 1], hspace=0.3, wspace=0.3)
        
        # CPU utilization subplot (larger)
        self.cpu_ax = self.fig.add_subplot(gs[0, 0])
        self.style_subplot(self.cpu_ax, 'CPU Utilization (%)')
        
        # Memory utilization subplot (larger)
        self.mem_ax = self.fig.add_subplot(gs[0, 1])
        self.style_subplot(self.mem_ax, 'Memory Utilization (%)')
        
        # Energy consumption subplot (full width)
        self.energy_ax = self.fig.add_subplot(gs[1, :])
        self.style_subplot(self.energy_ax, 'Energy Consumption (W)')
        
        # Create canvas that expands with window
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.visualization_frame)
        canvas_widget = self.canvas.get_tk_widget()
        canvas_widget.grid(row=0, column=0, sticky="nsew")
        
        # Add navigation toolbar in its own frame
        toolbar_frame = ttk.Frame(self.visualization_frame)
        toolbar_frame.grid(row=1, column=0, sticky="ew")
        NavigationToolbar2Tk(self.canvas, toolbar_frame)

    def style_subplot(self, ax, title):
        """Apply modern styling to subplot."""
        ax.set_facecolor(COLORS['graph_bg'])
        ax.set_title(title, color=COLORS['text'], pad=15, 
                    fontsize=12, fontweight='bold', fontfamily='Segoe UI')
        ax.set_xlabel('Time (s)', color=COLORS['text'], 
                     fontsize=10, labelpad=10, fontfamily='Segoe UI')
        ax.set_ylabel('Utilization (%)', color=COLORS['text'], 
                     fontsize=10, labelpad=10, fontfamily='Segoe UI')
        
        # Style grid
        ax.grid(True, linestyle='--', alpha=0.2, color=COLORS['grid'])
        ax.set_axisbelow(True)
        
        # Style spines
        for spine in ax.spines.values():
            spine.set_color(COLORS['border'])
            spine.set_linewidth(1)
        
        # Style ticks
        ax.tick_params(colors=COLORS['text'], which='both', 
                      labelsize=9, length=5, width=1)
        ax.set_ylim(0, 100)
        ax.set_xlim(0, 50)

    def setup_graphs(self):
        """Initialize the graphs with modern styling."""
        # Initialize data
        self.times = []
        self.cpu_data = []
        self.mem_data = []
        
        # Style both subplots
        for ax in [self.cpu_ax, self.mem_ax]:
            ax.set_facecolor(COLORS['background'])
            ax.grid(True, color=COLORS['grid'], linestyle='--', alpha=0.2)
            ax.tick_params(colors=COLORS['text'])
            for spine in ax.spines.values():
                spine.set_color(COLORS['border'])
                spine.set_linewidth(0.5)
        
        # Initialize lines with custom styling
        self.cpu_line, = self.cpu_ax.plot([], [], color=COLORS['cpu_line'], 
                                         linewidth=2.5, label='CPU Usage',
                                         marker='o', markersize=4)
        self.mem_line, = self.mem_ax.plot([], [], color=COLORS['mem_line'], 
                                         linewidth=2.5, label='Memory Usage',
                                         marker='o', markersize=4)
        
        # Add legends with custom styling
        for ax in [self.cpu_ax, self.mem_ax]:
            legend = ax.legend(facecolor=COLORS['background'], edgecolor=COLORS['border'],
                             fontsize=10, labelcolor=COLORS['text'])
            legend.get_frame().set_alpha(0.9)
        
        # Set initial limits
        for ax in [self.cpu_ax, self.mem_ax]:
            ax.set_ylim(0, 100)
            ax.set_xlim(0, 50)
            ax.grid(True, linestyle='--', alpha=0.3, color=COLORS['grid'])
            ax.set_axisbelow(True)  # Make grid appear behind the plot

    def start_graph_update(self):
        """Start periodic graph updates."""
        self.update_graphs()

    def update_graphs(self):
        """Update graphs with current utilization data."""
        if not hasattr(self, 'root'):
            return
            
        if self.simulation_running and self.hosts:
            current_time = len(self.times)
            self.times.append(current_time)
            
            # Calculate average utilization
            cpu_util = 0
            mem_util = 0
            active_hosts = [host for host in self.hosts if not host.is_failed]
            if active_hosts:
                for host in active_hosts:
                    cpu_util += (host.cpu_used / host.cpu_capacity * 100)
                    mem_util += (host.memory_used / host.memory_capacity * 100)
                cpu_util /= len(active_hosts)
                mem_util /= len(active_hosts)
            
            self.cpu_data.append(cpu_util)
            self.mem_data.append(mem_util)
            
            # Update line data
            self.cpu_line.set_data(self.times, self.cpu_data)
            self.mem_line.set_data(self.times, self.mem_data)
            
            # Dynamic x-axis adjustment with larger window
            if len(self.times) > 20:  # Reduced from 50 to 20 for faster updates
                for ax in [self.cpu_ax, self.mem_ax]:
                    ax.set_xlim(len(self.times) - 20, len(self.times))
            
            # Update canvas
            self.canvas.draw()
        
        # Schedule next update with faster interval
        if hasattr(self, 'root'):
            self.root.after(100, self.update_graphs)  # Reduced from 1000 to 100 ms

    def update_vm_table(self):
        """Update the VM table with current VM information."""
        # Clear existing items
        for item in self.vm_table.get_children():
            self.vm_table.delete(item)
        
        # Add current VMs
        for vm in self.vms:
            host_id = vm.host.id if vm.host else "None"
            status = vm.state
            time_info = f"{vm.execution_time:.1f}s" if vm.execution_time > 0 else "-"
            
            self.vm_table.insert("", "end", values=(
                vm.id,
                f"{vm.cpu_required} cores",
                f"{vm.memory_required} GB",
                vm.vm_type,
                vm.priority,
                host_id,
                status,
                time_info
            ))

    def start_simulation(self):
        """Start the simulation."""
        if not self.simulation_running:
            # Initialize simulation
            self.simulation_running = True
            self.simulation_paused = False
            
            # Load selected scenario if no VMs exist
            if not self.vms:
                self.load_scenario(self.scenario_var.get())
            
            # Initialize environment and scheduler
            self.env = simpy.Environment()
            algorithm = self.current_algorithm.get().lower().replace(" ", "_")
            quantum = float(self.quantum_var.get())
            self.scheduler = VMScheduler(self.env, self.hosts, self.vms, algorithm, quantum)
            self.scheduler.gui = self
            
            # Update button states
            self.start_button.configure(state=tk.DISABLED)
            self.pause_button.configure(state=tk.NORMAL)
            self.reset_button.configure(state=tk.DISABLED)
            
            # Start simulation thread
            self.simulation_thread = threading.Thread(target=self.run_simulation)
            self.simulation_thread.daemon = True
            self.simulation_thread.start()
            
            # Start updating displays
            self.update_displays()

    def update_displays(self):
        """Update all displays periodically."""
        if self.simulation_running and not self.simulation_paused:
            self.update_vm_table()
            self.update_graphs()
            
        if self.simulation_running:
            self.root.after(100, self.update_displays)

    def run_simulation(self):
        """Run the simulation in a separate thread."""
        try:
            # Run the simulation process
            self.env.process(self.scheduler.run())
            
            while self.simulation_running and not self.simulation_paused:
                try:
                    # Update simulation speed
                    self.simulation_speed = float(self.speed_scale.get())
                    
                    # Step the simulation environment
                    self.env.step()
                    
                    # Sleep based on simulation speed
                    time.sleep(0.1 / self.simulation_speed)
                    
                    # Check if simulation is complete
                    if self.scheduler.is_completed():
                        self.simulation_running = False
                        self.root.after(0, self.on_simulation_complete)
                        break
                        
                except ValueError:
                    # Use default speed if invalid
                    self.simulation_speed = DEFAULT_SIMULATION_SPEED
                    
        except Exception as e:
            print(f"Simulation error: {e}")
            self.simulation_running = False
            self.root.after(0, self.on_simulation_error, str(e))

    def pause_simulation(self):
        """Pause the simulation."""
        if self.simulation_running:
            self.simulation_paused = not self.simulation_paused
            button_text = "Resume" if self.simulation_paused else "Pause"
            self.pause_button.configure(text=button_text)
            self.reset_button.configure(state="normal")

    def reset_simulation(self):
        """Reset the simulation to initial state."""
        if self.simulation_thread and self.simulation_thread.is_alive():
            self.simulation_running = False
            self.simulation_thread.join()
        
        # Reset all VMs and hosts
        for vm in self.vms:
            vm.reset_state()
        for host in self.hosts:
            host.reset_state()
        
        # Clear history and graphs
        self.clear_graphs()
        
        # Reset GUI elements
        self.simulation_paused = False
        self.start_button.configure(text="Start", state="normal")
        self.pause_button.configure(state="disabled")
        self.reset_button.configure(state="disabled")
        
        # Update displays
        self.update_displays()

    def on_simulation_error(self, error_msg):
        """Handle simulation errors"""
        messagebox.showerror("Simulation Error", f"An error occurred during simulation:\n{error_msg}")
        self.reset_simulation()

    def on_simulation_complete(self):
        """Handle simulation completion"""
        self.start_button.configure(state=tk.NORMAL)
        self.pause_button.configure(state=tk.DISABLED)
        self.reset_button.configure(state=tk.NORMAL)
        
        # Show completion message
        messagebox.showinfo("Simulation Complete", 
                          "All VMs have completed their execution.")
        
        # Generate and show performance report
        self.performance_metrics.calculate_metrics(self.scheduler)
        report = self.performance_metrics.generate_report()
        
        # Create report window
        report_window = tk.Toplevel(self.root)
        report_window.title("Performance Report")
        report_window.geometry("400x300")
        
        report_text = tk.Text(report_window, wrap=tk.WORD)
        report_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        report_text.insert(tk.END, report)
        report_text.configure(state='disabled')

    def clear_graphs(self):
        """Clear all graphs."""
        self.times = []
        self.cpu_data = []
        self.mem_data = []
        
        for ax in [self.cpu_ax, self.mem_ax]:
            ax.clear()
            ax.set_ylim(0, 100)
            ax.set_xlim(0, 50)
            ax.grid(True, color=COLORS['grid'], linestyle='--', alpha=0.2)
        
        self.cpu_ax.set_title('CPU Utilization (%)', color=COLORS['text'])
        self.mem_ax.set_title('Memory Utilization (%)', color=COLORS['text'])
        
        # Reinitialize lines
        self.cpu_line, = self.cpu_ax.plot([], [], color=COLORS['cpu_line'], 
                                         linewidth=2.5, label='CPU Usage')
        self.mem_line, = self.mem_ax.plot([], [], color=COLORS['mem_line'], 
                                         linewidth=2.5, label='Memory Usage')
        
        self.canvas.draw()

    def load_scenario(self, scenario_name):
        """Load a predefined scenario"""
        scenario = SCENARIOS[scenario_name]
        
        # Clear existing VMs
        self.vms = []
        
        # Create VMs based on scenario
        for i in range(scenario['vm_count']):
            cpu = random.randint(*scenario['cpu_range'])
            memory = random.randint(*scenario['memory_range'])
            vm = VM(f"VM-{i+1}", cpu, memory)
            self.vms.append(vm)
        
        # Update display
        self.update_vm_table()

    def show_help(self):
        """Show help documentation"""
        help_window = tk.Toplevel(self.root)
        help_window.title("Help Documentation")
        help_window.geometry("600x400")
        
        # Create notebook for different sections
        notebook = ttk.Notebook(help_window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Overview tab
        overview_frame = ttk.Frame(notebook)
        notebook.add(overview_frame, text="Overview")
        
        overview_text = tk.Text(overview_frame, wrap=tk.WORD)
        overview_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        overview_text.insert(tk.END, """
Virtual Machine Scheduler Simulator
=================================

This simulator helps you understand and evaluate different VM scheduling algorithms in cloud environments.

Key Features:
• Multiple scheduling algorithms (First Fit, Round Robin, Priority, CFS)
• Real-time visualization of resource utilization
• Energy-aware scheduling
• Performance metrics and reporting
• Predefined test scenarios
• Interactive tutorials

Use the control panel to:
1. Select a scheduling algorithm
2. Choose a test scenario
3. Configure simulation parameters
4. Start/stop the simulation
5. View performance reports
""")
        overview_text.configure(state='disabled')
        
        # Algorithms tab
        algorithms_frame = ttk.Frame(notebook)
        notebook.add(algorithms_frame, text="Algorithms")
        
        algorithms_text = tk.Text(algorithms_frame, wrap=tk.WORD)
        algorithms_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        algorithms_text.insert(tk.END, """
Scheduling Algorithms
====================

1. First Fit
-----------
Assigns each VM to the first available host that can accommodate its resource requirements.
Best for: Simple, fast scheduling with moderate resource utilization.

2. Round Robin
-------------
Distributes VMs equally across available hosts in a circular manner.
Best for: Fair resource distribution in homogeneous environments.

3. Priority
----------
Schedules VMs based on their priority levels, with higher priority VMs getting resources first.
Best for: Environments with varying importance levels.

4. Completely Fair Scheduler (CFS)
--------------------------------
Ensures fair CPU time distribution by tracking virtual runtime.
Best for: Fair resource allocation in heterogeneous environments.
""")
        algorithms_text.configure(state='disabled')
        
        # Metrics tab
        metrics_frame = ttk.Frame(notebook)
        notebook.add(metrics_frame, text="Metrics")
        
        metrics_text = tk.Text(metrics_frame, wrap=tk.WORD)
        metrics_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        metrics_text.insert(tk.END, """
Performance Metrics
=================

1. Throughput
------------
Number of VMs completed per time unit.
Higher values indicate better performance.

2. Average Latency
----------------
Average time from VM creation to completion.
Lower values indicate better performance.

3. Fairness Index
----------------
Measures how evenly resources are distributed.
Values closer to 1 indicate better fairness.

4. Energy Efficiency
-------------------
Work done per unit of energy consumed.
Higher values indicate better energy efficiency.

5. Resource Utilization
----------------------
Percentage of CPU and memory resources used.
Optimal values depend on the scenario.
""")
        metrics_text.configure(state='disabled')

    def add_vm(self):
        """Add a new VM to the simulation."""
        try:
            # Get values from input fields
            cpu = int(self.vm_cpu_var.get())
            memory = int(self.vm_memory_var.get())
            vm_type = self.vm_type_var.get()
            priority = int(self.vm_priority_var.get()) if self.vm_priority_var.get() else 1
            
            # Create new VM
            vm_id = f"VM-{len(self.vms) + 1}"
            vm = VM(vm_id, cpu, memory, vm_type)
            vm.priority = priority
            
            # Add to list and update table
            self.vms.append(vm)
            self.update_vm_table()
            
            # Clear input fields
            self.vm_cpu_var.set("")
            self.vm_memory_var.set("")
            self.vm_priority_var.set("")
            
        except ValueError as e:
            messagebox.showerror("Error", "Please enter valid numeric values for CPU and memory.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to add VM: {str(e)}")

def main():
    root = tk.Tk()
    root.geometry("1200x800")
    root.configure(bg=COLORS['background'])
    root.title("Process Scheduler Simulator")
    
    # Set window icon (if available)
    try:
        root.iconbitmap("icon.ico")
    except:
        pass
    
    # Make window centered
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x = (screen_width - 1200) // 2
    y = (screen_height - 800) // 2
    root.geometry(f"1200x800+{x}+{y}")
    
    # Configure grid weights
    root.grid_rowconfigure(0, weight=1)
    root.grid_columnconfigure(0, weight=1)
    
    app = VMSchedulerGUI(root)
    app.grid(row=0, column=0, columnspan=1, sticky="nsew", padx=10, pady=10)
    
    root.mainloop()

if __name__ == "__main__":
    try:
        print("Python version:", sys.version)
        print("Tkinter version:", tk.TkVersion)
        main()
    except Exception as e:
        print("Error occurred:", str(e))
        import traceback
        traceback.print_exc()
