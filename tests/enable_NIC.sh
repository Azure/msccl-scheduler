set -x

echo 1 | sudo tee /sys/bus/pci/rescan
sleep 1
sudo ifconfig ib7 172.16.1.17 netmask 255.255.0.0 broadcast 0.0.0.0 up
sudo systemctl restart azure_persistent_rdma_naming.service