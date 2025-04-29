import os

from picsellia import Client, Experiment


def get_experiment() -> Experiment:
    if "api_token" not in os.environ:
        raise Exception("You must set an api_token to run this image")
    api_token = os.environ["api_token"]

    if "host" not in os.environ:
        host = "https://app.picsellia.com"
    else:
        host = os.environ["host"]

    if "organization_id" not in os.environ:
        organization_id = None
    else:
        organization_id = os.environ["organization_id"]

    client = Client(api_token=api_token, host=host, organization_id=organization_id)

    if "experiment_id" not in os.environ:
        raise Exception("You must set an experiment_id to run this image")
    experiment_id = os.environ["experiment_id"]
    experiment = client.get_experiment_by_id(id=experiment_id)

    return experiment
