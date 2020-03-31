# Parallelizing ImageNet Training with Metaflow 

Repository contains code to parallise training of Imagenet using Metaflow with Kubernetes GPU Support. To train ImageNet Pytorch's example code NN are directly used here. 

## Metaflow On Kubernetes Setup

- Instructions of Setup of Kubernetes GPU based cluster for Metaflow :
    - https://github.com/valayDave/metaflow-on-kubernetes-docs

- Using `@kube(cpu=4,memory=40000,gpu=4,image='anibali/pytorch:cuda-10.1')` will enable trainig on the GPU instance. 

- This code works Ideally for CPU and GPU setups. GPU jobs can fail If GPU Memory is not enough. This can be solved by increasing number of GPUS in decorator vs Decreasing Batchsize.Specially noticed with Resnet when Running this same flow with only 2 GPUs. 
- 

## TODO 
- [ ] Create Notebook to Show Results. 
- [ ] Document the Models that are trained using this script.
- [ ] 