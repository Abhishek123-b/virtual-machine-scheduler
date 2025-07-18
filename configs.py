from .colors import COLORS

# Style configuration for widgets
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

# Tooltips for UI elements
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

# Animation and simulation settings
ANIMATION_INTERVAL = 100  # ms
DEFAULT_SIMULATION_SPEED = 5.0

# VM States and Types
VM_STATES = {
    'PENDING': 'Pending',
    'RUNNING': 'Running',
    'MIGRATING': 'Migrating',
    'COMPLETED': 'Completed',
    'FAILED': 'Failed'
}

VM_TYPES = {
    'Standard': {'cpu_weight': 1.0, 'mem_weight': 1.0},
    'Compute': {'cpu_weight': 2.0, 'mem_weight': 0.5},
    'Memory': {'cpu_weight': 0.5, 'mem_weight': 2.0},
    'Balanced': {'cpu_weight': 1.5, 'mem_weight': 1.5}
}

# Energy model configuration
ENERGY_MODEL = {
    'idle_power': 50,     # Watts - reduced idle power for more realistic difference
    'max_power': 150,     # Watts - adjusted max power for better scaling
    'power_factor': 0.8,  # Efficiency factor - slightly reduced to account for overhead
}

ALGORITHM_INFO = {
    'FCFS': {
        'name': 'First Come First Serve',
        'description': 'Simple scheduling algorithm that allocates resources in order of arrival',
        'advantages': 'Simple to implement, fair for batch processing',
        'disadvantages': 'Not optimal for interactive workloads'
    },
    'Round Robin': {
        'name': 'Round Robin',
        'description': 'Allocates resources in a circular manner with fixed time quantum',
        'advantages': 'Fair distribution of resources',
        'disadvantages': 'May not be optimal for varying workloads'
    },
    'Priority': {
        'name': 'Priority-based',
        'description': 'Schedules VMs based on priority levels',
        'advantages': 'Supports different service levels',
        'disadvantages': 'May lead to starvation of low-priority VMs'
    },
    'CFS': {
        'name': 'Completely Fair Scheduler',
        'description': 'Maintains fairness using virtual runtime',
        'advantages': 'Excellent fairness, good for mixed workloads',
        'disadvantages': 'More complex implementation'
    }
} 