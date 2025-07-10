from argparse import ArgumentParser

from orjson import orjson
from picsellia import Client
from picsellia.types.enums import ProcessingType


def create_processing(
    client: Client,
    name: str,
    type: str | ProcessingType,
    default_cpu: int,
    default_gpu: int,
    default_parameters: dict,
    docker_image: str,
    docker_tag: str,
    docker_flags: list[str] | None = None,
) -> str:
    payload = {
        "name": name,
        "type": type,
        "default_cpu": default_cpu,
        "default_gpu": default_gpu,
        "default_parameters": default_parameters,
        "docker_image": docker_image,
        "docker_tag": docker_tag,
        "docker_flags": docker_flags,
    }
    r = client.connexion.post(
        f"/sdk/organization/{client.id}/processings", data=orjson.dumps(payload)
    ).json()
    return r["id"]


if __name__ == "__main__":
    parser = ArgumentParser("Create a processing")
    parser.add_argument("--api_token", type=str)
    parser.add_argument("--organization_id", type=str)
    parser.add_argument("--host", type=str)
    parser.add_argument("--processing_name", type=str)
    parser.add_argument("--processing_type", type=str)
    parser.add_argument("--default_cpu", type=int)
    parser.add_argument("--default_gpu", type=int)
    parser.add_argument("--docker_image", type=str)
    parser.add_argument("--docker_tag", type=str)
    parser.add_argument("--docker_flags", nargs="+", type=str, default=None)
    args = parser.parse_args()

    client = Client(
        api_token=args.api_token, organization_id=args.organization_id, host=args.host
    )

    default_parameters = {
        "tags_list": ["is_woman", "is_man"],
        "device": "cuda:0",
        "batch_size": 8,
    }

    create_processing(
        client=client,
        name=args.processing_name,
        type=args.processing_type,
        default_cpu=args.default_cpu,
        default_gpu=args.default_gpu,
        default_parameters=default_parameters,
        docker_image=args.docker_image,
        docker_tag=args.docker_tag,
        docker_flags=args.docker_flags,
    )
