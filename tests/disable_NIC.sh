echo "Disabling 1 NIC..."
set -x
echo 1 | sudo tee /sys/bus/pci/devices/0108:00:00.0/remove
