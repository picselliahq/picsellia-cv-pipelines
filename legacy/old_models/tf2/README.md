

[![Picsell.ia](https://i.ibb.co/4N8XyQ0/rsz-11rsz-picsellia.png)](https://www.picsellia.com)

# Welcome to Picsell.ia Public Repo !
Whether you're an AI enthusiast, a top researcher or an AI Start-up building your product. Learn how to leverage Picsell.ia Platform to speed up your AI creation process.

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

This repo is a soft wrapper around Tensorflow Object Detection API

Use the 3 Notebooks in Colab or locally to freely:

  - Train an Object Detection Model with Transfer Learning
  - Train an Object segmentation Model with Transfer Learning
  - Train a classification Model with Tensorflow 2 and Transfer Learning
  - Send results and checkpoints to our back to version everything

To launch training within a container, run the following command :
  docker run --rm -d -e api_token=<api_token> -e experiment_id=<experiment_id> -e project_token=<project_token> --gpus all --name training picsellpn/trainingtf2:capture
