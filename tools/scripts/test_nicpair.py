import select
import subprocess
import sys
import time
import os
    
nic_1 = sys.argv[1]
nic_2 = sys.argv[2]
worldsize = sys.argv[3]
hostlist = sys.argv[4]
pair_key = f"{nic_1}-{nic_2}"


print("==============Testing NIC pair: ", nic_1, nic_2, "================")
command = "mpirun"
args = [
        "--mca", "btl_tcp_if_include", "eth0",
        "--mca", "pml", "ob1",
        "-mca", "btl", "^openib",
        "--allow-run-as-root",
        "--bind-to", "numa",
        "--tag-output",
        "-np", worldsize,
        "-npernode", "1",
        "-H", hostlist,
        "-x", "CUDA_VISIBLE_DEVICES={}".format(nic_1),
        "python", "/root/msccl/scheduler/msccl-scheduler/tools/scripts/NIC_mpi4py.py",
        str(nic_1),
        str(nic_2)
    ]
whole_command = [command] + args
env=os.environ
new_env = {k: v for k, v in env.items() if "MPI" not in k and "PMIX" not in k and "NCCL" not in k}
# with open(f'output_{nic_1}.txt', 'w') as f:
#     for k, v in new_env.items():
#         f.write(f'{k}={v}\n')
process = subprocess.Popen(whole_command, shell=False, env=new_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
print(whole_command)

poll_obj = select.poll()
poll_obj.register(process.stdout, select.POLLIN)

timeout = 10000
start_time = time.time()
max_duration = 60

while True:
    if time.time() - start_time > max_duration:  
        print("Execution Time Exceeded")
        process.terminate() 
        break
    
    poll_result = poll_obj.poll(timeout)
    if poll_result:
        line = process.stdout.readline().strip()
        if line:
            print(line)  # Print output dynamically
            if "No device found" in line:
                process.terminate()
        else:
            print("Execution Complete")
            break
    else:
        print("Execution Timeout")
        process.terminate()
        break