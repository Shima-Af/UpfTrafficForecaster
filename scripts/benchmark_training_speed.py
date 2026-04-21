import time, torch, numpy as np, yaml, pandas as pd
from torch.utils.data import DataLoader
from src.dataset import build_graph_datasets
from src.models.stgnn import TrafficSTGNN

params = yaml.safe_load(open("params.yaml"))
params["training"]["batch_size"] = 4

voronoi_map = pd.read_parquet("data/graphs/voronoi_map.parquet").squeeze()
voronoi_map.index.name = None
node_index_df = pd.read_parquet("data/graphs/node_index.parquet")
edge_index = torch.from_numpy(np.load("data/graphs/edge_index.npy")).cuda()

print("Building datasets...")
t0 = time.time()
train_ds, val_ds, test_ds, scaler, feature_cols, n_nodes = build_graph_datasets(
    "data/netmob/processed", params, voronoi_map, node_index_df
)
print(f"Dataset build: {time.time()-t0:.1f}s  |  train={len(train_ds)}  val={len(val_ds)}")

model = TrafficSTGNN.from_params(params, n_nodes=n_nodes, n_features=len(feature_cols)).cuda()
loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=0)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

model.train()
times = []
for i, (X, y) in enumerate(loader):
    X, y = X.cuda(), y.cuda()
    t0 = time.time()
    optimizer.zero_grad()
    pred = model(X, edge_index)
    loss = criterion(pred, y)
    loss.backward()
    optimizer.step()
    elapsed = time.time() - t0
    if i >= 2:
        times.append(elapsed)
        print(f"  batch {i}: {elapsed*1000:.0f} ms  loss={loss.item():.6f}")
    if i >= 11:
        break

n_train_batches = len(train_ds) // 4
n_val_batches   = len(val_ds)   // 4
avg_train = sum(times) / len(times)
epoch_s = avg_train * n_train_batches + (avg_train * 0.5) * n_val_batches
print(f"\nAvg step (train): {avg_train*1000:.0f} ms")
print(f"Batches/epoch — train: {n_train_batches}  val: {n_val_batches}")
print(f"Estimated epoch:  {epoch_s/60:.1f} min")
print(f"Estimated 100 epochs: {epoch_s*100/3600:.1f} h  (with early stopping likely fewer)")
