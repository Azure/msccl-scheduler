from mpi4py import MPI
import pynvml
import socket
import torch
import torch.distributed as dist
import os
import sys 
    
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
pynvml.nvmlInit()

if rank < 8:
    os.environ["NCCL_IB_HCA"] = 'mlx5_ib'+ sys.argv[1]
else:
    os.environ["NCCL_IB_HCA"] = 'mlx5_ib' + sys.argv[2]

if rank == 0:
    hostname = socket.gethostname()
    master_ip = socket.gethostbyname(hostname)
    master_port = "808" + sys.argv[1]
else:
    master_ip=""
    master_port=""

master_ip = comm.bcast(master_ip, root=0)
master_port = comm.bcast(master_port, root=0)

print("rank:" + str(rank))
print("wordsize:" + str(size))

# os.environ["RANK"] = str(rank)
# os.environ["OMPI_COMM_WORLD_RANK"] = str(rank)
# os.environ["WORLD_SIZE"] = str(size)
os.environ["MASTER_ADDR"] = master_ip
os.environ["MASTER_PORT"] = master_port
os.environ['NCCL_DEBUG'] = 'INFO'
os.environ['NCCL_SOCKET_IFNAME'] = "eth0"
os.environ['NCCL_P2P_DISABLE'] = "1"
os.environ['NCCL_SHM_DISABLE'] = "1"


print("master_ip:" + master_ip)
print("master_port:" + master_port)

dist.init_process_group(backend='nccl', init_method='env://', world_size=size, rank=rank)

tensor_size = 10  
input_tensor = torch.ones(tensor_size).cuda(0)
output_tensor =  [torch.zeros(tensor_size).cuda(0) for _ in range(size)]

print(input_tensor)
print(output_tensor)

dist.all_gather(output_tensor, input_tensor)


print(input_tensor)
print(output_tensor)

MPI.Finalize()
pynvml.nvmlShutdown()