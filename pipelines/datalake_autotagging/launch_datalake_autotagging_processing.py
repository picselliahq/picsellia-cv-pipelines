import os
from argparse import ArgumentParser
from uuid import UUID

import orjson
from picsellia import Client, Datalake, Job


def get_processing(client: Client, name: str) -> str:
    r = client.connexion.get(
        f"/sdk/organization/{client.id}/processings", params={"name": name}
    ).json()
    return r["items"][0]["id"]


def launch_processing(
    client: Client,
    datalake: Datalake,
    data_ids: list[UUID],
    model_version_id: str,
    processing_id: str,
    parameters: dict,
    cpu: int,
    gpu: int,
    target_datalake_name: str | None = None,
):
    payload = {
        "processing_id": processing_id,
        "parameters": parameters,
        "cpu": cpu,
        "gpu": gpu,
        "model_version_id": model_version_id,
        "data_ids": data_ids,
    }

    # If given, create a new datalake
    if target_datalake_name:
        payload["target_datalake_name"] = target_datalake_name

    r = client.connexion.post(
        f"/api/datalake/{datalake.id}/processing/launch",
        data=orjson.dumps(payload),
    ).json()
    return Job(client.connexion, r, version=2)


if __name__ == "__main__":
    parser = ArgumentParser("Launch a processing")
    parser.add_argument("--api_token", type=str)
    parser.add_argument("--organization_id", type=str)
    parser.add_argument("--host", type=str)
    parser.add_argument("--processing_name", type=str)
    parser.add_argument("--datalake_id", type=str)
    parser.add_argument("--target_datalake_name", type=str, default=None)
    parser.add_argument("--model_version_id", type=str)
    parser.add_argument("--cpu", type=int, default=8)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--data_ids", nargs="+", type=str)
    parser.add_argument("--tags_list", nargs="+", type=str)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    client = Client(
        api_token=args.api_token, organization_id=args.organization_id, host=args.host
    )

    data_ids = [UUID(data_id) for data_id in args.data_ids]

    datalake = client.get_datalake(id=args.datalake_id)

    processing_id = get_processing(client, args.processing_name)

    job = launch_processing(
        client=client,
        datalake=datalake,
        data_ids=data_ids,
        model_version_id=args.model_version_id,
        processing_id=processing_id,
        parameters={
            "tags_list": args.tags_list,
            "device": args.device,
            "batch_size": args.batch_size,
        },
        cpu=args.cpu,
        gpu=args.gpu,
        target_datalake_name=args.target_datalake_name,
    )

    sync_job = job.sync()
    print(f"sync job: {sync_job}")

    os.environ["job_id"] = str(job.id)
