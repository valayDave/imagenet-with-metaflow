# Parallelizing ImageNet Training with Metaflow 

Repository contains code to parallise training of Imagenet using Metaflow with Kubernetes GPU Support. To train ImageNet Pytorch's example code NN are directly used here. 

## Metaflow On Kubernetes Setup

- Instructions of Setup of Kubernetes GPU based cluster for Metaflow :
    - https://github.com/valayDave/metaflow-on-kubernetes-docs

- Using `@kube(cpu=4,memory=40000,gpu=4,image='anibali/pytorch:cuda-10.1')` will enable GPU based training on cluster.  

- This code works for CPU and GPU setups. GPU jobs can fail If GPU Memory is not enough. This can be solved by increasing number of GPUS in decorator vs Decreasing Batchsize.Specially noticed with Resnet when Running this same flow with only 2 GPUs. 

- Download Dataset From https://tiny-imagenet.herokuapp.com/

- S3 Required for Parallelised GPU training on AWS. 

## Running Instructions
1. Upload dataset to S3 and change parameter `dataset_s3_path` in the `imagenet_metaflow.py` or add it into your run parameters. 
2. To Run on Local, Comment out `@kube` decorator and use the below command: 
    ```
    python imagenet_metaflow.py --metadata local run --max-workers 2 
    ```

## TODO 
- [ ] Create Notebook to Show Results. 
- [ ] Document the Models that are trained using this script.
- [ ] Create Hyper Param Experiments for Model training. 
- [ ] Run Model training in Parallel for 10 models Over More than 30+ Epochs. Training Takes lot of time for ResNet. Need Bigger Machines
