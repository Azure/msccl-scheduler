set -x

echo 1 | sudo tee /sys/bus/pci/rescan
sleep 1
sudo systemctl restart azure_persistent_rdma_naming.service