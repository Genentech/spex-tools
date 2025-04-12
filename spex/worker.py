from multiprocessing import Process
import time

class Worker(Process):
    def __init__(self, event_queue, name="Worker"):
        super().__init__(daemon=True)
        self.event_queue = event_queue
        self.name = name

    def run(self):
        print(f"{self.name} started.")
        while True:
            for event in self.event_queue.consume():
                print(f"{self.name} processing event: {event}")
                time.sleep(1)  # simulate work
            time.sleep(0.5)
