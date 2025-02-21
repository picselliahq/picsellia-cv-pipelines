import subprocess
import shlex
from typing import List

from picsellia import Client
import os
import re
import json

os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["PICSELLIA_SDK_CUSTOM_LOGGING"] = "True"
os.environ["PICSELLIA_SDK_DOWNLOAD_BAR_MODE"] = "2"
os.environ["PICSELLIA_SDK_SECTION_HANDLER"] = "1"

from datetime import datetime
from picsellia.types.enums import JobRunStatus
import logging

logging.getLogger("picsellia").setLevel(logging.INFO)
os.chdir("tf")
command = "python3 main.py"

if "host" not in os.environ:
    host = "https://app.picsellia.com"
else:
    host = os.environ["host"]
if "api_token" not in os.environ:
    raise RuntimeError("You must set an api_token to run this image")

if "job_id" not in os.environ:
    raise RuntimeError("No job_id found in env. variables")

api_token = os.environ["api_token"]
job_id = os.environ["job_id"]

client = Client(api_token=api_token, host=host)

job = client.get_job_by_id(job_id)
job.update_job_run_with_status(JobRunStatus.RUNNING)
process = subprocess.Popen(
    shlex.split(command), stdout=subprocess.PIPE, stderr=subprocess.STDOUT
)
part = "--#--Start processing"
replace_log = False
buffer: List[str] = []
start_buffer = False
buffer_length = 0
try:
    job.send_logging(part, part)
except Exception:
    pass
logs = {}
logs[part] = {"datetime": str(datetime.now().isoformat()), "logs": {}}
last_line = ""
while True:
    if not process.stdout:
        break
    output = process.stdout.readline()
    if output.decode("utf-8") == "" and process.poll() is not None:
        break
    text = output.decode("utf-8")
    if output:
        if text.startswith("--#--"):
            part = output.decode("utf-8")
            logs[part] = {"datetime": str(datetime.now().isoformat()), "logs": {}}
        if text.startswith("-----"):
            progress_line_nb = job.line_nb
            replace_log = True

        if text.startswith("--*--"):
            replace_log = False

        if re.match("--[0-9]--", text[:6]):
            start_buffer = True
            buffer_length = int(text[2])

        if re.match("---[0-9]---", text[:8]):
            start_buffer = False
            try:
                job.send_logging(buffer, part, special="buffer")
                job.line_nb += len(buffer) - 1
            except Exception:
                pass
            buffer = []

        if start_buffer:
            buffer.append(text)
            logs[part]["logs"][str(job.line_nb + len(buffer))] = text
            if len(buffer) == buffer_length:
                try:
                    job.send_logging(buffer, part, special="buffer")
                    job.line_nb += buffer_length - 1
                except Exception:
                    pass
                buffer = []
        else:
            if not replace_log:
                try:
                    job.send_logging(text, part)
                    logs[part]["logs"][str(job.line_nb)] = text
                except Exception:
                    pass
            else:
                try:
                    job.line_nb = progress_line_nb
                    job.send_logging(text, part)
                except Exception:
                    pass

        last_line = text

logs_path = "{}-logs.json".format(job_id)
with open(logs_path, "w") as f:
    if buffer != []:
        for i, line in enumerate(buffer):
            logs[part]["logs"][str(job.line_nb + i)] = line
        job.send_logging(buffer, part, special="buffer")
    logs["exit_code"] = {
        "exit_code": str(process.returncode),
        "datetime": str(datetime.now().isoformat()),
    }
    json.dump(logs, f)
job.send_logging(str(process.returncode), part, special="exit_code")
job.store_logging_file(logs_path)

if process.returncode == 0 or process.returncode == "0":
    job.update_job_run_with_status(JobRunStatus.SUCCEEDED)
else:
    job.update_job_run_with_status(JobRunStatus.FAILED)
rc = process.poll()
