from torch_geometric.loader import DataLoader

from PatientGraphDataset import PatientGraphDataset

'''
# DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

for batch in train_loader:
    # batch.x -> Feature di tutti i nodi del batch
    # batch.edge_index -> Archi aggiornati per il batch
    # batch.batch -> Vettore che indica a quale paziente appartiene ogni nodo
    out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
    # loss = ...
'''


# Inizializza il dataset (passa il path dove hai salvato i .pt)
dataset = PatientGraphDataset(root='data_graphs_processed')

# Shuffle e Split (80% train, 20% test)
dataset = dataset.shuffle()
train_dataset = dataset[:int(len(dataset) * 0.8)]
test_dataset = dataset[int(len(dataset) * 0.8):]

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print()
print(f'Dataset: {dataset}:')
print('====================')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')

print()
print(dataset[0])
print('=============================================================')

# Gather some statistics about the first graph.
print(f'Number of nodes: {dataset[0].num_nodes}')
print(f'Number of edges: {dataset[0].num_edges}')
print(f'Average node degree: {dataset[0].num_edges / dataset[0].num_nodes:.2f}')
print(f'Has isolated nodes: {dataset[0].has_isolated_nodes()}')
print(f'Has self-loops: {dataset[0].has_self_loops()}')
print(f'Is undirected: {dataset[0].is_undirected()}')  # TODO non risulta undirected ma in teoria lo è, è un errore interno, anche facendo delle prove con ToUndirected() continuava a dare False



'''
# train loop

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CancerGNN(num_node_features=7, num_edge_features=3, hidden_channels=64).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()


def train():
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()

        # Passiamo x, edge_index, edge_attr e il vettore batch
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)

        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(train_dataset)


def test(loader):
    model.eval()
    correct = 0
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        pred = out.argmax(dim=1)
        correct += int((pred == data.y).sum())
    return correct / len(loader.dataset)


for epoch in range(1, 101):
    loss = train()
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
'''