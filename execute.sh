#! /bin/sh

python simulation_epsilon_acc.py -nt 10000 -np 1000 -dis 5
python simulation_epsilon_acc.py -nt 10000 -np 50 -dis 4
python simulation_np_acc.py -np 50 -dis 4
python simulation_nq_acc.py -np 50 -dis 4
python simulation_lamda.py
python simulation_depth.py
python simulation_partition.py
python simulation_range_parameter.py
python realdata.py -method all