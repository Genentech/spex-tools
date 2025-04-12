# Hello Worker

Isolated Python library with a simple worker system.

## Installation

```bash
pip install .
```

## Usage

```python
from hello_worker import say_hello, Worker, EventQueue

# Simple function
print(say_hello("World"))

# Event system
event_queue = EventQueue()
event_queue.publish("Hello Event 1")
event_queue.publish("Hello Event 2")

# Start worker
worker = Worker(event_queue, name="MyWorker")
worker.start()

# Wait a bit to process events
import time
time.sleep(3)
```
