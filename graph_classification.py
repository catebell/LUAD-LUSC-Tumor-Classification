import torch
from torch_geometric.loader import DataLoader

from PatientGraphDataset import PatientGraphDataset
from models.GCN import GCN
from models.CancerGNN import CancerGNN


dataset = PatientGraphDataset(root='data_graphs_processed')  # dataset initialization; if not exists, it gets created

print()
print(f'Dataset: {dataset}:')
print('====================')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')

print(dataset[0])
print('=============================================================')

# Gather some statistics about the first graph.
print(f'Number of nodes: {dataset[0].num_nodes}')
print(f'Number of edges: {dataset[0].num_edges}')
print(f'Average node degree: {dataset[0].num_edges / dataset[0].num_nodes:.2f}')
print(f'Has isolated nodes: {dataset[0].has_isolated_nodes()}')
print(f'Has self-loops: {dataset[0].has_self_loops()}')
print(f'Is undirected: {dataset[0].is_undirected()}')  # TODO non risulta undirected ma in teoria lo è, è un errore interno, anche facendo delle prove con ToUndirected() continuava a dare False


# (80% train, 20% test)
torch.manual_seed(12345)
dataset = dataset.shuffle()
train_dataset = dataset[:int(len(dataset) * 0.8)]
test_dataset = dataset[int(len(dataset) * 0.8):]

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

'''
for step, data in enumerate(train_loader):
    print(f'\nStep {step + 1}:')
    print('=======')
    print(f'Number of graphs in the current batch: {data.num_graphs}')
    print(data)
'''


# train loop

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#model = GCN(num_node_features=5, num_classes=2, hidden_channels=64).to(device)
model = CancerGNN(num_node_features=5, num_edge_features=3, hidden_channels=64).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # lr = Learning Rate

criterion = torch.nn.CrossEntropyLoss()

print(model)


def train():
    model.train()
    total_loss = 0

    for data in train_loader:  # Iterate in batches over the training dataset.
        #out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)

        loss = criterion(out, data.y)  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(train_dataset)


def test(loader):
     model.eval()
     correct = 0

     for data in loader:  # Iterate in batches over the training/test dataset.
         #out = model(data.x, data.edge_index, data.batch)
         out = model(data.x, data.edge_index, data.edge_attr, data.batch)
         pred = out.argmax(dim=1)  # Use the class with the highest probability.
         correct += int((pred == data.y).sum())  # Check against ground-truth labels.
     return correct / len(loader.dataset)  # Derive ratio of correct predictions.


for epoch in range(1, 101):
    loss = train()
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
