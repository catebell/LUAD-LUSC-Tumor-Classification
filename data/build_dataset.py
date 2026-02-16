import torch
from torch_geometric.data import Data
from clinical_data_preprocessing import process_clinical_data


# =========================
# FINTA funzione grafo
# =========================

def load_graph_for_patient(case_id):
    """
    Simula caricamento grafo per paziente.
    Sostituirai con implementazione reale.
    """

    num_genes = 1000
    num_edges = 4000

    node_features = torch.randn(num_genes, 3)
    edge_index = torch.randint(0, num_genes, (2, num_edges))

    return node_features, edge_index


# =========================
# Build dataset completo
# =========================

def build_full_dataset():

    (
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        train_ids, val_ids, test_ids
    ) = process_clinical_data()

    # Convert to torch
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_val   = torch.tensor(X_val, dtype=torch.float32)
    X_test  = torch.tensor(X_test, dtype=torch.float32)

    y_train = torch.tensor(y_train, dtype=torch.long)
    y_val   = torch.tensor(y_val, dtype=torch.long)
    y_test  = torch.tensor(y_test, dtype=torch.long)

    # =========================
    # TRAIN
    # =========================

    train_dataset = []

    for i, case_id in enumerate(train_ids):

        node_features, edge_index = load_graph_for_patient(case_id)

        data = Data(
            x=node_features,
            edge_index=edge_index,
            clinical=X_train[i],
            y=y_train[i]
        )

        train_dataset.append(data)

    # =========================
    # VAL
    # =========================

    val_dataset = []

    for i, case_id in enumerate(val_ids):

        node_features, edge_index = load_graph_for_patient(case_id)

        data = Data(
            x=node_features,
            edge_index=edge_index,
            clinical=X_val[i],
            y=y_val[i]
        )

        val_dataset.append(data)

    # =========================
    # TEST
    # =========================

    test_dataset = []

    for i, case_id in enumerate(test_ids):

        node_features, edge_index = load_graph_for_patient(case_id)

        data = Data(
            x=node_features,
            edge_index=edge_index,
            clinical=X_test[i],
            y=y_test[i]
        )

        test_dataset.append(data)

    # =========================
    # SAVE
    # =========================

    torch.save(train_dataset, "train_dataset.pt")
    torch.save(val_dataset, "val_dataset.pt")
    torch.save(test_dataset, "test_dataset.pt")

    print("Dataset (clinico + grafo) salvati correttamente.")


if __name__ == "__main__":
    build_full_dataset()