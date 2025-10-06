from ast import boolop
from pydantic import BaseModel
import uvicorn
import requests
import time
import os
import json
import glob
from fastapi import FastAPI
import threading

class RegisterWorkerRequest(BaseModel):
    worker_url: str

class RegisterWorkerReply(BaseModel):
    worker_id: int

class GetTaskRequest(BaseModel):
    worker_id: int

class HeartbeatRequest(BaseModel):
    worker_id: int

class GetTaskReply(BaseModel):
    job_id: int
    task_id: int
    file: str
    output_dir: str
    map_fn: str
    reduce: bool
    wait: bool
    n_reduce: int
    fn_kwargs: dict
    job_exists: bool

class FinishTaskRequest(BaseModel):
    job_id: int
    task_id: int
    reduce: bool
    success: bool

COORDINATOR_URL = "http://localhost:8000"
WAIT_TIME_S = 1
MAP_DIR = "data/mapreduce/tmp-map"

def find_open_port():
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port

def char_count(filename: str):
    char_counts = {}
    with open(filename, "r") as f:
        for line in f:
            for char in line:
                if char in char_counts:
                    char_counts[char] += 1
                else:
                    char_counts[char] = 1
    return char_counts
    

def map_fn(map_fn_name: str, filename: str, map_task_id: int, n_reduce: int, **map_fn_kwargs):
    reduce_bucket = hash(filename) % n_reduce
    os.makedirs(MAP_DIR, exist_ok=True)
    output_filename = os.path.join(MAP_DIR, f"mr-{map_task_id}-{reduce_bucket}.json")
    if map_fn_name == "cc":
        output = char_count(filename)
        with open(output_filename, "w") as f:
            json.dump(output, f)

def reduce_fn(reduce_id: int, output_dir: str):
    import glob
    pattern = os.path.join(MAP_DIR, f"mr-*-{reduce_id}.json")
    files = glob.glob(pattern)
    
    reduced_counts = {}
    for file in files:
        with open(file, "r") as f:
            data = json.load(f)
            for key, value in data.items():
                if key in reduced_counts:
                    reduced_counts[key] += value
                else:
                    reduced_counts[key] = value
    
    with open(os.path.join(output_dir, f"reduce-{reduce_id}.json"), "w") as f:
        json.dump(reduced_counts, f)

    return reduced_counts

class Worker:
    def __init__(self, worker_port):
        self.port = worker_port
        self.worker_id = None
        self.register_worker()
        
        self.app = FastAPI()
        self.heartbeat_thread = threading.Thread(
            target=self.heartbeat_loop, daemon=True
        )
        self.heartbeat_thread.start()
        self.worker_loop()

    def register_worker(self):
        request = RegisterWorkerRequest(worker_url=f"http://localhost:{self.port}")
        response = requests.post(f"{COORDINATOR_URL}/register_worker", json=request.model_dump())
        task_reply = RegisterWorkerReply.model_validate(response.json())
        self.worker_id = task_reply.worker_id

    def get_work_from_coordinator(self):
        request = GetTaskRequest(worker_id=self.worker_id)
        response = requests.post(f"{COORDINATOR_URL}/get_work", json=request.model_dump())
        task_reply = GetTaskReply.model_validate(response.json())
        return task_reply
    
    def worker_loop(self):
        while True:
            task = None
            try:
                task = self.get_work_from_coordinator()
            except Exception as e:
                continue
            
            if task and task.job_exists:
                if not task.reduce:
                    map_fn(
                        map_fn_name=task.map_fn,
                        filename=task.file,
                        map_task_id=task.task_id,
                        n_reduce=task.n_reduce,
                    )
                else:
                    reduce_fn(
                        reduce_id=task.task_id,
                        output_dir=task.output_dir,
                    )
                request = FinishTaskRequest(
                    job_id=task.job_id,
                    task_id=task.task_id,
                    reduce=task.reduce,
                    success=True,
                )
                response = requests.post(f"{COORDINATOR_URL}/finish_work", json=request.model_dump())

            time.sleep(WAIT_TIME_S)
        
    def heartbeat_loop(self):
        while True:
            request = HeartbeatRequest(worker_id=self.worker_id)
            response = requests.post(f"{COORDINATOR_URL}/heartbeat", json=request.model_dump())
            time.sleep(5)


if __name__ == "__main__":
    worker_port = find_open_port()
    worker = Worker(worker_port)
    uvicorn.run(worker.app, host="0.0.0.0", port=worker_port)