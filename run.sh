# multiple GPUs
NGPUS=2
python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/train_sg_net.py --config-file "sgg_configs/vg_vrd/rel_danfeiX_FPN50_imp_nobias_albert.yaml" 