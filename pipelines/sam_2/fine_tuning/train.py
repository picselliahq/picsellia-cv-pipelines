# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import random
import sys
import traceback
from argparse import ArgumentParser

import submitit
import torch
from hydra import compose, initialize_config_module
from hydra.utils import instantiate
from iopath.common.file_io import g_pathmgr
from omegaconf import OmegaConf
from training.utils.train_utils import makedir, register_omegaconf_resolvers

os.environ["HYDRA_FULL_ERROR"] = "1"


def single_proc_run(local_rank, main_port, cfg, world_size):
    """Single GPU process"""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(main_port)
    os.environ["RANK"] = str(local_rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    try:
        register_omegaconf_resolvers()
    except Exception as e:
        logging.info(e)

    trainer = instantiate(cfg.trainer, _recursive_=False)
    trainer.run()


def single_node_runner(cfg, main_port: int):
    assert cfg.launcher.num_nodes == 1
    num_proc = cfg.launcher.gpus_per_node
    torch.multiprocessing.set_start_method(
        "spawn"
    )  # CUDA runtime does not support `fork`
    if num_proc == 1:
        # directly call single_proc so we can easily set breakpoints
        # mp.spawn does not let us set breakpoints
        single_proc_run(local_rank=0, main_port=main_port, cfg=cfg, world_size=num_proc)
    else:
        mp_runner = torch.multiprocessing.start_processes
        args = (main_port, cfg, num_proc)
        # Note: using "fork" below, "spawn" causes time and error regressions. Using
        # spawn changes the default multiprocessing context to spawn, which doesn't
        # interact well with the dataloaders (likely due to the use of OpenCV).
        mp_runner(single_proc_run, args=args, nprocs=num_proc, start_method="spawn")


def format_exception(e: Exception, limit=20):
    traceback_str = "".join(traceback.format_tb(e.__traceback__, limit=limit))
    return f"{type(e).__name__}: {e}\nTraceback:\n{traceback_str}"


class SubmititRunner(submitit.helpers.Checkpointable):
    """A callable which is passed to submitit to launch the jobs."""

    def __init__(self, port, cfg):
        self.cfg = cfg
        self.port = port
        self.has_setup = False

    def run_trainer(self):
        job_env = submitit.JobEnvironment()
        # Need to add this again so the hydra.job.set_env PYTHONPATH
        # is also set when launching jobs.
        add_pythonpath_to_sys_path()
        os.environ["MASTER_ADDR"] = job_env.hostnames[0]
        os.environ["MASTER_PORT"] = str(self.port)
        os.environ["RANK"] = str(job_env.global_rank)
        os.environ["LOCAL_RANK"] = str(job_env.local_rank)
        os.environ["WORLD_SIZE"] = str(job_env.num_tasks)

        register_omegaconf_resolvers()
        cfg_resolved = OmegaConf.to_container(self.cfg, resolve=False)
        cfg_resolved = OmegaConf.create(cfg_resolved)

        trainer = instantiate(cfg_resolved.trainer, _recursive_=False)
        trainer.run()

    def __call__(self):
        job_env = submitit.JobEnvironment()
        self.setup_job_info(job_env.job_id, job_env.global_rank)
        try:
            self.run_trainer()
        except Exception as e:
            # Log the exception. Then raise it again (as what SubmititRunner currently does).
            message = format_exception(e)
            logging.error(message)
            raise e

    def setup_job_info(self, job_id, rank):
        """Set up slurm job info"""
        self.job_info = {
            "job_id": job_id,
            "rank": rank,
            "cluster": self.cfg.get("cluster", None),
            "experiment_log_dir": self.cfg.launcher.experiment_log_dir,
        }

        self.has_setup = True


def add_pythonpath_to_sys_path():
    if "PYTHONPATH" not in os.environ or not os.environ["PYTHONPATH"]:
        return
    sys.path = os.environ["PYTHONPATH"].split(":") + sys.path


def parse_args_and_compose_config(args):
    cfg = compose(config_name=args.config, overrides=args.overrides)
    if cfg.launcher.experiment_log_dir is None:
        cfg.launcher.experiment_log_dir = os.path.join(
            os.getcwd(), "sam2_logs", args.config
        )
    print("###################### Train App Config ####################")
    print(OmegaConf.to_yaml(cfg))
    print("############################################################")
    add_pythonpath_to_sys_path()
    makedir(cfg.launcher.experiment_log_dir)
    return cfg


def save_config_files(cfg, config_name):
    cfg_resolved = OmegaConf.to_container(cfg, resolve=False)
    cfg_resolved = OmegaConf.create(cfg_resolved)

    with g_pathmgr.open(
        os.path.join(cfg.launcher.experiment_log_dir, "config.yaml"), "w"
    ) as f:
        f.write(OmegaConf.to_yaml(cfg))

    with g_pathmgr.open(
        os.path.join(cfg.launcher.experiment_log_dir, "config_resolved.yaml"), "w"
    ) as f:
        f.write(OmegaConf.to_yaml(cfg_resolved, resolve=True))


def run_on_cluster(cfg, args, submitit_conf, submitit_dir):
    executor = submitit.AutoExecutor(folder=submitit_dir)

    submitit_conf.partition = args.partition or submitit_conf.get("partition", None)
    submitit_conf.account = args.account or submitit_conf.get("account", None)
    submitit_conf.qos = args.qos or submitit_conf.get("qos", None)

    job_kwargs = {
        "timeout_min": 60 * submitit_conf.timeout_hour,
        "name": getattr(submitit_conf, "name", args.config),
        "slurm_partition": submitit_conf.partition,
        "gpus_per_node": cfg.launcher.gpus_per_node,
        "tasks_per_node": cfg.launcher.gpus_per_node,
        "cpus_per_task": submitit_conf.cpus_per_task,
        "nodes": cfg.launcher.num_nodes,
        "slurm_additional_parameters": {
            "exclude": " ".join(submitit_conf.get("exclude_nodes", [])),
        },
    }

    if "include_nodes" in submitit_conf:
        assert len(submitit_conf["include_nodes"]) >= cfg.launcher.num_nodes, (
            "Not enough nodes"
        )
        job_kwargs["slurm_additional_parameters"]["nodelist"] = " ".join(
            submitit_conf["include_nodes"]
        )

    if submitit_conf.account:
        job_kwargs["slurm_additional_parameters"]["account"] = submitit_conf.account
    if submitit_conf.qos:
        job_kwargs["slurm_additional_parameters"]["qos"] = submitit_conf.qos
    if submitit_conf.get("mem_gb"):
        job_kwargs["mem_gb"] = submitit_conf.mem_gb
    elif submitit_conf.get("mem"):
        job_kwargs["slurm_mem"] = submitit_conf.mem
    if submitit_conf.get("constraints"):
        job_kwargs["slurm_constraint"] = submitit_conf.constraints
    if submitit_conf.get("comment"):
        job_kwargs["slurm_comment"] = submitit_conf.comment
    if submitit_conf.get("srun_args", {}).get("cpu_bind"):
        job_kwargs["slurm_srun_args"] = ["--cpu-bind", submitit_conf.srun_args.cpu_bind]

    print("###################### SLURM Config ####################")
    print(job_kwargs)
    print("########################################################")
    executor.update_parameters(**job_kwargs)

    main_port = random.randint(*submitit_conf.port_range)
    runner = SubmititRunner(main_port, cfg)
    job = executor.submit(runner)
    print(f"Submitit Job ID: {job.job_id}")
    runner.setup_job_info(job.job_id, rank=0)


def launch_training_job(cfg, args):
    submitit_conf = cfg.get("submitit", None)
    assert submitit_conf is not None, "Missing submitit config"

    submitit_dir = os.path.join(cfg.launcher.experiment_log_dir, "submitit_logs")
    cfg.launcher.gpus_per_node = (
        args.num_gpus if args.num_gpus is not None else cfg.launcher.gpus_per_node
    )
    cfg.launcher.num_nodes = (
        args.num_nodes if args.num_nodes is not None else cfg.launcher.num_nodes
    )
    submitit_conf.use_cluster = (
        args.use_cluster if args.use_cluster is not None else submitit_conf.use_cluster
    )

    if submitit_conf.use_cluster:
        run_on_cluster(cfg, args, submitit_conf, submitit_dir)
    else:
        cfg.launcher.num_nodes = 1
        main_port = random.randint(*submitit_conf.port_range)
        single_node_runner(cfg, main_port)


def main(args) -> None:
    cfg = parse_args_and_compose_config(args)
    save_config_files(cfg, args.config)
    launch_training_job(cfg, args)


if __name__ == "__main__":
    initialize_config_module("sam2", version_base="1.2")
    parser = ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        required=True,
        type=str,
        help="path to config file (e.g. configs/sam2.1_training/sam2.1_hiera_b+_MOSE_finetune.yaml)",
    )
    parser.add_argument(
        "--use-cluster",
        type=int,
        default=None,
        help="whether to launch on a cluster, 0: run locally, 1: run on a cluster",
    )
    parser.add_argument("--partition", type=str, default=None, help="SLURM partition")
    parser.add_argument("--account", type=str, default=None, help="SLURM account")
    parser.add_argument("--qos", type=str, default=None, help="SLURM qos")
    parser.add_argument(
        "--num-gpus", type=int, default=None, help="number of GPUS per node"
    )
    parser.add_argument("--num-nodes", type=int, default=None, help="Number of nodes")
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Overrides to pass to Hydra config (e.g. scratch.train_batch_size=8)",
    )
    args = parser.parse_args()
    args.use_cluster = bool(args.use_cluster) if args.use_cluster is not None else None
    register_omegaconf_resolvers()
    main(args)
