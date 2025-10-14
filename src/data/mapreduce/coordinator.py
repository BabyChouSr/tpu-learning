from fastapi import FastAPI
from data.mapreduce.client import SubmitJobRequest, PollJobRequest, PollJobReply, SubmitJobReply
from data.mapreduce.worker import GetTaskReply, RegisterWorkerRequest, RegisterWorkerReply, GetTaskRequest, FinishTaskRequest, HeartbeatRequest
from pydantic import BaseModel
from dataclasses import dataclass
import logging
import uvicorn
import time
import threading

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class JobStatus:
    done: bool
    failed: bool

@dataclass
class TaskStatus:
    done: bool
    failed: bool

@dataclass
class TaskInfo:
    task_id: int
    task_request: GetTaskReply
    done: bool
    failed: bool

@dataclass
class WorkerInfo:
    worker_id: int
    last_heartbeat_time: int
    worker_url: str
    tasks: list[TaskInfo]

@dataclass
class JobInfo:
    job_request: SubmitJobRequest
    job_status: JobStatus
    map_task_infos: list[TaskInfo]
    reduce_task_infos: list[TaskInfo]

HEARTBEAT_EXPIRATION_SECONDS = 30

class Coordinator:
    def __init__(self):
        # State management
        self.num_workers = 0
        self.num_jobs = 0
        self.workers: dict[int, WorkerInfo] = {}
        self.job_id_to_job_info: dict[int, JobInfo] = {}
        self.task_queue = []
        self.heartbeat_thread = threading.Thread(
            target=self.remove_stale_worker_by_expiration, 
        )
        self.task_id_to_worker_id: dict[int, int] = {}
        self.heartbeat_thread.start()

        # FastAPI wrangling
        self.app = FastAPI()
        self.app.post("/submit", response_model=SubmitJobReply)(self.submit)
        self.app.post("/register_worker", response_model=RegisterWorkerReply)(self.register_worker)
        self.app.post("/get_job_status", response_model=PollJobReply)(self.get_job_status)
        self.app.post("/get_work", response_model=GetTaskReply)(self.assign_work)
        self.app.post("/finish_work")(self.finish_worker_task)
        self.app.post("/heartbeat")(self.receive_heartbeat)

    async def register_worker(self, request: RegisterWorkerRequest):
        worker_id = self.num_workers
        self.workers[self.num_workers] = WorkerInfo(
            worker_id=worker_id,
            last_heartbeat_time=time.time(),
            worker_url=request.worker_url,
            tasks=[],
        )
        logger.info(f"Received request from {request.worker_url} with id: {self.num_workers}")
        self.num_workers += 1
        return RegisterWorkerReply(worker_id=worker_id)

    async def submit(self, request: SubmitJobRequest):
        logger.info(f"Received request from client: {request}")
        job_id = self.num_jobs

        map_task_infos = []
        for idx, in_file in enumerate(request.files):
            map_task_info = TaskInfo(
                task_id=idx,
                task_request=GetTaskReply(
                    job_id=job_id,
                    task_id=idx,
                    file=in_file,
                    output_dir=request.output_dir,
                    map_fn=request.map_fn,
                    reduce=False,
                    wait=False,
                    fn_kwargs=request.fn_kwargs,
                    n_reduce=request.n_reduce,
                    job_exists=True,
                ),
                done=False,
                failed=False,
            )
            map_task_infos.append(map_task_info)
            self.task_queue.append(map_task_info)
    
        reduce_task_infos = []
        for idx in range(request.n_reduce):
            reduce_task_info = TaskInfo(
                task_id=idx,
                task_request=GetTaskReply(
                    job_id=job_id,
                    task_id=idx,
                    file="not needed",
                    output_dir=request.output_dir,
                    map_fn="reduce", # NOTE(chris) not implemented yet
                    reduce=True,
                    wait=True,
                    fn_kwargs=request.fn_kwargs, # NOTE(chris) reduction kwargs not implemented
                    n_reduce=request.n_reduce,
                    job_exists=True,
                ),
                done=False,
                failed=False,
            )
            reduce_task_infos.append(reduce_task_info)
            self.task_queue.append(reduce_task_info)
            
        self.job_id_to_job_info[job_id] = JobInfo(
            job_request=request,
            job_status=JobStatus(done=False, failed=False),
            map_task_infos=map_task_infos,
            reduce_task_infos=reduce_task_infos,
        )

        self.num_jobs += 1
        return SubmitJobReply(job_id=job_id)

    async def get_job_status(self, request: PollJobRequest):
        logger.info(f"Received job status request for {request.job_id}")
        job_info = self.job_id_to_job_info[request.job_id]
        return PollJobReply(done=job_info.job_status.done, failed=job_info.job_status.failed)

    async def assign_work(self, request: GetTaskRequest):
        worker_id = request.worker_id
        if len(self.task_queue) == 0:
            logger.info("Task queue is empty, not scheduling any tasks.")
            return GetTaskReply(
                job_id=-1,
                task_id=-1,
                file="-1",
                output_dir="",
                map_fn="NA", # NOTE(chris) not implemented yet
                reduce=False,
                wait=False,
                fn_kwargs={}, # NOTE(chris) reduction kwargs not implemented
                n_reduce=-1,
                job_exists=False,
            )
        else:
            task_to_schedule_idx = None
            task_idx = 0
            while task_idx < len(self.task_queue) and task_to_schedule_idx is None:
                task = self.task_queue[task_idx]
                if task.task_request.wait == False: # Can instantly schedule
                    logger.info(f"Scheduling task: {task.task_id} for job: {task.task_request.job_id} to worker {worker_id}")
                    task_to_schedule_idx = task_idx

                task_idx += 1
                
            if task_to_schedule_idx is not None:
                task_to_schedule = self.task_queue[task_to_schedule_idx]
                self.workers[worker_id].tasks.append(task_to_schedule)
                self.task_queue.pop(task_to_schedule_idx)
                return task_to_schedule.task_request
            else:
                logger.info(f"Could not schedule a task at the moment")
                return {} # No task is possible to run

    async def finish_worker_task(self, request: FinishTaskRequest):
        job_info = self.job_id_to_job_info[request.job_id]
        if request.reduce:
            task_infos = job_info.reduce_task_infos
        else:
            task_infos = job_info.map_task_infos

        for task_info in task_infos:
            if task_info.task_id == request.task_id:
                if request.success:
                    logger.info(f"Finished task {request.task_id} for job {request.job_id}")
                    task_info.done = True
                    task_info.failed = False
                else:
                    task_info.failed = True

        all_map_tasks_status = []
        for map_task_info in job_info.map_task_infos:
            all_map_tasks_status.append(map_task_info.done)
        all_map_tasks_done = all(all_map_tasks_status)
        if all_map_tasks_done: # schedule the reduce task
            for reduce_task in job_info.reduce_task_infos:
                reduce_task.task_request.wait = False

    async def receive_heartbeat(self, request: HeartbeatRequest):
        worker_id = request.worker_id
        self.workers[worker_id].last_heartbeat_time = time.time()
        logger.info(f"Received heartbeat from worker {worker_id} at time {time.time()}")

    def remove_stale_worker_by_expiration(self):
        while True:
            alive_workers = {}
            for worker_id, worker in self.workers.items():
                if time.time() - worker.last_heartbeat_time > HEARTBEAT_EXPIRATION_SECONDS:
                    logger.info(f"Noticed that worker {worker_id} has died")
                    for task in worker.tasks: # Requeue the tasks
                        if not task.done:
                            self.task_queue.insert(0, task)
                else:
                    alive_workers[worker_id] = worker
        
            self.workers = alive_workers
            time.sleep(1)

if __name__ == "__main__":
    coordinator = Coordinator()
    uvicorn.run(coordinator.app, host="0.0.0.0", port=8000)