# Predefined test scenarios
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

# Tutorial information
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

# Scheduling algorithm steps
SCHEDULING_STEPS = {
    'fcfs': [
        ('Queue Check', 'Checking pending VMs in order of arrival'),
        ('Resource Check', 'Verifying available resources on hosts'),
        ('VM Allocation', 'Allocating VMs to available hosts'),
        ('Status Update', 'Updating VM and host status')
    ],
    'round robin': [
        ('Time Slice', 'Setting up time quantum for each VM'),
        ('Queue Rotation', 'Rotating through VM queue'),
        ('Resource Check', 'Verifying available resources'),
        ('VM Migration', 'Moving VMs between hosts')
    ],
    'priority': [
        ('Priority Sort', 'Sorting VMs by priority level'),
        ('Resource Check', 'Checking resource availability'),
        ('VM Allocation', 'Allocating high-priority VMs first'),
        ('Queue Update', 'Updating waiting queue')
    ],
    'cfs': [
        ('Virtual Runtime', 'Calculating virtual runtime for VMs'),
        ('Fairness Check', 'Ensuring fair resource distribution'),
        ('Load Balance', 'Balancing load across hosts'),
        ('State Update', 'Updating system state')
    ]
} 