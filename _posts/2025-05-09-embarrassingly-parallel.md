---
layout: post
author: Mark Hobbs
title: Scaling embarrassingly parallel processes in the cloud
draft: True
---

```python
import os
import uuid
import docker
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

class ModelManager:
    def __init__(self, image_name: str, n_workers: int = 4, base_io_dir: str = "/tmp/model_manager_io"):
        self.image_name = image_name
        self.n_workers = n_workers
        self.client = docker.from_env()
        self.base_io_dir = Path(base_io_dir)
        self.containers = []

        self.base_io_dir.mkdir(parents=True, exist_ok=True)

    def _run_container_with_io(self, input_data: str, index: int):
        """Runs one container with isolated input and output."""
        job_id = f"job_{index}_{uuid.uuid4().hex[:6]}"
        job_dir = self.base_io_dir / job_id
        input_file = job_dir / "input.txt"
        output_file = job_dir / "output.txt"

        job_dir.mkdir(parents=True, exist_ok=True)
        input_file.write_text(input_data)

        container = self.client.containers.run(
            self.image_name,
            command=f"python run_model.py /data/input.txt /data/output.txt",  # Adjust as needed
            volumes={str(job_dir): {'bind': '/data', 'mode': 'rw'}},
            detach=True,
            auto_remove=True
        )
        return container, job_dir

    def run_parallel(self, input_list: list[str]):
        """Run containers with per-instance input in parallel."""
        if len(input_list) > self.n_workers:
            raise ValueError("More input sets than available workers")

        def task(i, input_data):
            return self._run_container_with_io(input_data, i)

        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            futures = [executor.submit(task, i, input_data) for i, input_data in enumerate(input_list)]
            results = [f.result() for f in futures]
            self.containers, self.job_dirs = zip(*results)

    def collect_outputs(self):
        """Collect outputs from the finished containers."""
        outputs = []
        for container, job_dir in zip(self.containers, self.job_dirs):
            container.wait()
            output_file = job_dir / "output.txt"
            if output_file.exists():
                outputs.append(output_file.read_text())
            else:
                outputs.append(None)
        return outputs

    def cleanup(self):
        for container in self.containers:
            try:
                container.stop()
                container.remove()
            except Exception:
                pass
        # Optional: remove I/O directories
        # shutil.rmtree(self.base_io_dir)
```

**Example usage:**

```python
manager = ModelManager("my_model_image", n_workers=4)

# Inputs to be sent to 4 workers
inputs = [
    "param1=10\nparam2=20",
    "param1=15\nparam2=25",
    "param1=20\nparam2=30",
    "param1=25\nparam2=35",
]

manager.run_parallel(inputs)
results = manager.collect_outputs()
manager.cleanup()

for i, output in enumerate(results):
    print(f"Output from worker {i}:\n{output}")
```

