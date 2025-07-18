# Virtual Machine Scheduler Simulator

A comprehensive simulator for evaluating different VM scheduling algorithms in cloud environments.

## Features

- Multiple scheduling algorithms:
  - Round Robin
  - Best Fit
  - Energy-Aware
  - Priority-Based
  - Workload-Aware
- Real-time visualization of:
  - VM allocation success rate
  - Host utilization
  - VM migrations
  - Energy consumption
- Support for different workload types:
  - Normal
  - CPU-intensive
  - Memory-intensive
  - Bursty
- Performance metrics collection and export
- Interactive GUI for simulation control

## Requirements

- Python 3.8+
- Dependencies listed in `requirements.txt`

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/vm-scheduler-simulator.git
cd vm-scheduler-simulator
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the simulator:
```bash
python src/gui/main_window.py
```

2. Configure simulation parameters:
   - Number of hosts
   - Number of VMs
   - Scheduling algorithm
   - Workload type

3. Start the simulation and observe:
   - Real-time resource utilization
   - VM allocation and migration patterns
   - Energy consumption
   - Performance metrics

4. Export metrics:
   - Metrics are automatically exported to `simulation_metrics.csv` when simulation stops

## Project Structure

```
vm-scheduler-simulator/
├── src/
│   ├── gui/
│   │   └── main_window.py
│   ├── models/
│   │   ├── host.py
│   │   ├── vm.py
│   │   ├── workload_generator.py
│   │   └── metrics.py
│   └── schedulers/
│       └── scheduler.py
├── requirements.txt
└── README.md
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This project was developed as part of a cloud computing research initiative
- Special thanks to the open-source community for their valuable tools and libraries 