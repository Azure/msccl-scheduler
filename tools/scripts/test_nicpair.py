import subprocess
import sys
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
new_env = {k: v for k, v in env.items() if"MPI" not in k}
process = subprocess.Popen(whole_command, shell=False, env=new_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
print(whole_command)

output = ""
try:
    while process.poll() is None:
        line = process.stdout.readline().strip()
        if line:
            output += line + "\n"
            print(line)  # Print output dynamically
            if "No device found" in line:
                process.terminate()
                print(pair_key)  # Return the failed pair key
    process.wait()
    if process.returncode != 0:
        print("Subprocess returned non-zero exit code:", process.returncode)
except subprocess.TimeoutExpired:
    process.terminate()
    print("Timeout occurred for NIC pair: ", nic_1, nic_2)
    print(pair_key)  # Return the failed pair key