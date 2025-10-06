from pydantic import BaseModel
from typing import Any
import requests

class SubmitJobRequest(BaseModel):
    files: list[str]
    output_dir: str
    map_fn: str
    n_reduce: int
    fn_kwargs: dict

class SubmitJobReply(BaseModel):
    job_id: int

class PollJobRequest(BaseModel):
    job_id: int

class PollJobReply(BaseModel):
    done: bool
    failed: bool

COORDINATOR_URL = "http://localhost:8000/"

def main():
    request = SubmitJobRequest(
        files=["data/samples/mapreduce/input/a.txt", "data/samples/mapreduce/input/b.txt"],
        output_dir="data/samples/mapreduce/output",
        map_fn="cc",
        n_reduce=2,
        fn_kwargs={"x": 1}
    )
    response = requests.post(f"{COORDINATOR_URL}/submit", json=request.model_dump())
    submit_reply = SubmitJobReply.model_validate(response.json())
    job_id = submit_reply.job_id

    request = PollJobRequest(
        job_id=job_id
    )
    response = requests.post(f"{COORDINATOR_URL}/get_job_status", json=request.model_dump())
    print(response.json())


if __name__ == "__main__":
    main()    
