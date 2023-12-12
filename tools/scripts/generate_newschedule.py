import argparse
import networkx as nx
import pipeline_allgather
import to_xml
import xml.etree.ElementTree as ET
from xml.dom import minidom

PCIE_BW = 23.5
NVLINK_BW = 256

def topology(nnode: int):
    G = nx.DiGraph()
    compute_nodes = []
    for a in range(nnode):
        for b in range(8):
            compute_nodes.append((a, "GPU", b))
            # NVSwitch
            G.add_edge((a, "GPU", b), (a, "NVSwitch", 0), capacity=NVLINK_BW)
            G.add_edge((a, "NVSwitch", 0), (a, "GPU", b), capacity=NVLINK_BW)
            # PCIe Swtich
            G.add_edge((a, "GPU", b), (a, "PCIe Switch", b // 2), capacity=PCIE_BW)
            G.add_edge((a, "PCIe Switch", b // 2), (a, "GPU", b), capacity=PCIE_BW)
        # mlx5 nics
        for b in range(4):
            G.add_edge((a, "PCIe Switch", b), (a, "mlx5", b * 2), capacity=PCIE_BW)
            G.add_edge((a, "PCIe Switch", b), (a, "mlx5", b * 2 + 1), capacity=PCIE_BW)
            G.add_edge((a, "mlx5", b * 2), (a, "PCIe Switch", b), capacity=PCIE_BW)
            G.add_edge((a, "mlx5", b * 2 + 1), (a, "PCIe Switch", b), capacity=PCIE_BW)
        # IB Switch: separate IBs for odd/even channel mlx5
        for b in range(8):
            G.add_edge((-1, "IB", b % 2), (a, "mlx5", b), capacity=PCIE_BW)
            G.add_edge((a, "mlx5", b), (-1, "IB", b % 2), capacity=PCIE_BW)
    return G, compute_nodes


k_value = 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='default')
    parser.add_argument('fileName', type=str, help='the output file name')
    parser.add_argument('nodes', type=int, help='number of nodes in the cluster', default='2')
    parser.add_argument('inplace', type=int, help='whether use inplace during data transfer', default='0')
    args = parser.parse_args()
    
    topo, compute_nodes = topology(args.nodes)
    (U, k), (Ts, Cs) = pipeline_allgather.optimal_pipeline_spanning_trees(
        topo, compute_nodes=compute_nodes, fixed_K=k_value)
    # print(f"U={U}, k={k}")
    # print(f"{len(compute_nodes) * k / U} GBps")

    gpu_index = lambda n: n[0] * 8 + n[2]
    nTs = {}
    for (u, i), ps in Ts.items():
        nps = []
        nTs[gpu_index(u), i] = nps
        for p in ps:
            ns = set([n for n, _ in p] + [n for _, n in p])
            assert not ((-1, "IB", 0) in ns and (-1, "IB", 1) in ns)
            odd_even = None
            if (-1, "IB", 0) in ns:
                odd_even = 0
            elif (-1, "IB", 1) in ns:
                odd_even = 1
            nps.append([(gpu_index(p[0][0]), gpu_index(p[-1][-1]), odd_even)])
    nCs = {(gpu_index(u), i): c for (u, i), c in Cs.items()}
    nodes = sorted(map(gpu_index, compute_nodes))

    nics = {a: [] for a in range(args.nodes)}
    for a, tp, b in topo.nodes():
        if tp == 'mlx5':
            nics[a].append(b)
    nics_str = "\n".join(f"node {a} mlx5 nics {' '.join(map(str, sorted(bs)))}" for a, bs in sorted(nics.items()))
    print(nics_str) 

    algo = to_xml.construct_algo_allreduce(nTs, nCs, k, nodes, 1, inplace=args.inplace, nics_str=nics_str)
    s = ET.tostring(algo, 'utf-8')
    s = minidom.parseString(s)
    s = s.toprettyxml(indent="  ")
    s = '\n'.join(s.split('\n')[1:])
    with open(args.fileName, 'w') as f:
        f.write(s)
