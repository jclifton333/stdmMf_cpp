"""Plot networks for use in paper"""

import os
import sys

root_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),"..")
sys.path.append(os.path.join(root_dir, "build/src/main/"))
import network_pb2

import igraph

import argparse

import re

def form_adjacency(node_list):
    num_nodes = len(node_list.nodes)
    adj = [ [0] * num_nodes for i in range(num_nodes)]
    coords = []
    for node in node_list.nodes:
        if node.HasField("x") and node.HasField("y"):
            coords.append((node.x, node.y))

        for neigh in node.neigh:
            adj[node.index][neigh] = 1

    assert len(coords) == 0 or len(coords) == num_nodes

    if len(coords) == 0:
        return adj, None
    else:
        return adj, coords

def plot_network(node_list, save_name):
    adj, layout = form_adjacency(node_list)

    graph = igraph.Graph.Adjacency(adj).as_undirected()

    if layout is None:
        igraph.plot(graph, layout = "fr",
                    vertex_size = 4, vertex_color = "#000000",
                    target = save_name)
    else:
        igraph.plot(graph, layout = layout,
                    vertex_size = 4, vertex_color = "#000000",
                    target = save_name)


def main(data_dir):
    net_check = re.compile("(grid_[0-9]+x[0-9]+\.pb"
                           "|barabasi_[0-9]+\.pb"
                           "|random_[0-9]+\.pb)")

    net_files = []
    for file_name in os.listdir(data_dir):
        if net_check.match(file_name):
            net_files.append(file_name)


    for net_file in net_files:
        node_list = network_pb2.NodeList()
        with open(os.path.join(data_dir, net_file), "r") as f:
            node_list.ParseFromString(f.read())

        save_name = os.path.join(data_dir,
                                 os.path.splitext(net_file)[0] + ".png")
        plot_network(node_list, save_name)
        print "saved %s" % os.path.basename(save_name)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type = str, required = True)

    args = ap.parse_args()

    main(args.data_dir)
