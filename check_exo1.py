import torch
from learning.tsp.dataset import load_dataset
from model.goal import GOAL
from utils.data_manipulation import *

torch.manual_seed(0)
net = GOAL(1,128,9,512, "relu", 8, 4, "relu", 8)

batch = next(iter(load_dataset("data/atsp100_test.npz", 4, 4, what="test")))
raw = prepare_batch(batch, "tsp", "cpu", sample=False)
node_f, edge_f, pdata = prepare_data(raw, "tsp")
#node_features (4, 101, 2)
#edge_features = None

net.eval()
with torch.no_grad():
    scores = net(node_f, edge_f, pdata )
#scores  attendu (4, 101)

net.train()  # un pas d'entrainement sur un sous-probleme de 20 noeuds
node_f, edge_f, pdata = prepare_data([raw[0][:, :20, :20], None], "tsp")
loss = torch.nn.functional.cross_entropy(net(node_f, edge_f, pdata), create_ground_truth(4, pdata, "cpu"))
loss.backward()
print("perte                 :", round(loss.item(), 3), "(finie, proche de 2.9 au hasard)")
print("grad node_adapter.tsp :", net.node_adapter.tsp.grad is not None, "(attendu True)")
print("grad edge_adapter.tsp :", net.edge_adapter.tsp.grad, "(attendu None)")
print("grad lambda_z1        :", net.layers[0].attn.lambda_z1.grad, "(attendu None)")