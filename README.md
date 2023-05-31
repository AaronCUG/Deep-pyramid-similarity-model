# Deep-pyramid-similarity-model
This is the code for Deep pyramid similarity model. 

Requirements: python >= 3.6, pytorch >= 1.0

Testing command (Ubuntu): python -m torch.distributed.launch --nproc_per_node=4 DPSM.py --imgpairfile .../Sample.csv

where --nproc_per_node is the number of GPUs; --imgpairfile should be assigned to a local file path of Sample.csv.

building_patches and masks are two test input datasets. Both should be unzipped into the same folder with Sample.csv.
