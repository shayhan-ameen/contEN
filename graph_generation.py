__author__ = 'Shayhan'

__author__ = 'Shayhan'

import networkx as nx
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F


# def generate_graph(crf_df, patients_data, D_0_TIME):
#     patients_events = create_events(crf_df, patients_data, D_0_TIME)
#     patients_events_df = convert_to_dataframe(patients_events)
#     patients_events_df = patients_events_df.dropna()
#     scaler = MinMaxScaler() # Initialize the scaler
#     numeric_cols = [col for col in patients_events_df.columns if col != 'pt_index'] # Select numeric columns for scaling
#     patients_events_df[numeric_cols] = scaler.fit_transform(patients_events_df[numeric_cols]) # Apply scaling

# def patient_graph_generation(crf_df, patients_data):
#     for pt in crf_df.index:
#         pt_data = patients_data[pt]
#         bgl_indices = np.nonzero(pt_data['ens'][D_0_TIME:D_1_TIME])[0]

def create_events(crf_df, patients_data, D_0_TIME, min_num_nodes=1):
    patients_events = []
    patient_without_graph = []
    for pt in crf_df.index:
        patient_events = []
        pt_data = patients_data[pt]
        bgl_indices = np.nonzero(pt_data['bst'][:D_0_TIME])[0]
        # print(bgl_indices)
        # print(len(bgl_indices))
        # if len(bgl_indices) == 0:
        #     event = {'pt_index': np.nan}
        #     patient_events.append(event)
        if len(bgl_indices) >=min_num_nodes :
            for i, bgl_idx in enumerate(bgl_indices[1:], start=1):
                # print(i, bgl_idx)
                interval_start = bgl_indices[i - 1]
                interval_end = bgl_indices[i]
                event = {
                    'pt_index': pt,
                    'bgl': pt_data['bst'][interval_end],
                    'pre_bgl': pt_data['bst'][interval_start],
                    'insulin': pt_data['insulin'][interval_start:interval_end].sum(),
                    'calorie': pt_data['calorie'][interval_start:interval_end].sum(),
                    'duration': interval_end - interval_start
                }
                patient_events.append(event)
                patients_events.append(patient_events)
        else:
            patient_without_graph.append(pt)
    return patients_events, patient_without_graph

# def create_events(crf_df, patients_data, D_0_TIME):
#     patients_events = []
#     for pt in crf_df.index:
#         patient_events = []
#         pt_data = patients_data[pt]
#         bgl_indices = np.nonzero(pt_data['bst'][:D_0_TIME])[0]
#         # print(bgl_indices)
#         # print(len(bgl_indices))
#         if len(bgl_indices) == 0:
#             event = {'pt_index': np.nan}
#             patient_events.append(event)
#         else:
#             for i, bgl_idx in enumerate(bgl_indices[1:], start=1):
#                 # print(i, bgl_idx)
#                 interval_start = bgl_indices[i - 1]
#                 interval_end = bgl_indices[i]
#                 event = {
#                     'pt_index': pt,
#                     'bgl': pt_data['bst'][interval_end],
#                     'pre_bgl': pt_data['bst'][interval_start],
#                     'insulin': pt_data['insulin'][interval_start:interval_end].sum(),
#                     'calorie': pt_data['calorie'][interval_start:interval_end].sum(),
#                     'duration': interval_end - interval_start
#                 }
#                 patient_events.append(event)
#         patients_events.append(patient_events)
#     return patients_events


def convert_to_dataframe(patients_events):
    flat_events = [event for patient in patients_events for event in patient]
    return pd.DataFrame(flat_events)


def create_graph_data_from_events(events):
    node_features = []
    edge_index = []
    num_features = 5

    for i, (_, event) in enumerate(events.iterrows()):
        feat = [
            float(event.get('bgl', 0.0)),
            float(event.get('pre_bgl', 0.0)),
            float(event.get('insulin', 0.0)),
            float(event.get('calorie', 0.0)),
            float(event.get('duration', 0.0))
        ]
        node_features.append(feat)
        # edge_index.append([i, i])
        if i > 0:
            edge_index.append([i - 1, i])

    # edge_index, _ = add_self_loops(edge_index)

    if not node_features:
        x = torch.zeros((1, num_features), dtype=torch.float)
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        x = torch.tensor(node_features, dtype=torch.float)

        if edge_index:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

            # max_index = x.shape[0] - 1
            # edge_index = edge_index[:, (edge_index < max_index).all(dim=0)]  # **Fixed condition**
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

    return Data(x=x, edge_index=edge_index)

def create_all_patient_graphs(crf_df, patients_data, D_0_TIME, min_num_nodes=1):
    patients_events, patient_without_graph = create_events(crf_df, patients_data, D_0_TIME, min_num_nodes=min_num_nodes)
    patients_events_df = convert_to_dataframe(patients_events)
    # patients_events_df = patients_events_df.dropna()
    patients_events_df = patients_events_df.dropna().reset_index(drop=True)  # **CHANGE THIS LINE**

    scaler = MinMaxScaler()  # Initialize the scaler
    numeric_cols = [col for col in patients_events_df.columns if
                    col != 'pt_index']  # Select numeric columns for scaling
    patients_events_df[numeric_cols] = scaler.fit_transform(patients_events_df[numeric_cols])

    all_patient_graph_list = {}
    for pt_index in patients_events_df['pt_index'].unique():
        events = patients_events_df[patients_events_df['pt_index'] == pt_index]
        if events.empty:
            print(f"$$ During EVENT to DATAFRAME no enent fournt for {pt_index}")
            patient_without_graph.append(pt_index)
        #     print("Patient Index = {pt_index} do not contain any GRAPH.------------------------------------------")
        #     graph_data = -1
        #     all_patient_graph_list[pt_index] = graph_data
        else:
            graph_data = create_graph_data_from_events(events)
            all_patient_graph_list[pt_index] = graph_data
    return all_patient_graph_list, patient_without_graph

# def create_all_patient_graphs(patients_events_df):
#     all_patient_graph_list = {}
#     for pt_index in patients_events_df['pt_index'].unique():
#         events = patients_events_df[patients_events_df['pt_index'] == pt_index]
#         if events.empty:
#             print("Patient Index = {pt_index} do not contain any GRAPH.------------------------------------------")
#             graph_data = -1
#             all_patient_graph_list[pt_index] = graph_data
#         else:
#             graph_data = create_graph_data_from_events(events)
#             all_patient_graph_list[pt_index] = graph_data
#     return all_patient_graph_list


def create_batch_from_patients_events(all_patient_graph_list, flt_patients):
    batch_data_list = []
    for pt in flt_patients:
        batch_data_list.append(all_patient_graph_list[pt])
    batch_data = Batch.from_data_list(batch_data_list)
    return batch_data


# def create_batch_from_patients_events(patients_events_df, train_patients, test_patients):
#     train_data_list = []
#     test_data_list = []
#
#     for pt_index in train_patients:
#         events = patients_events_df[patients_events_df['pt_index'] == pt_index]
#         if events.empty: print("Patient {pt_index} do not contain any GRAPH.------------------------------------------")
#         data = create_graph_data_from_events(events)
#         train_data_list.append(data)
#
#     for pt_index in test_patients:
#         events = patients_events_df[patients_events_df['pt_index'] == pt_index]
#         if events.empty: print("Patient {pt_index} do not contain any GRAPH.------------------------------------------")
#         data = create_graph_data_from_events(events)
#         test_data_list.append(data)
#
#     # Convert to PyG Batch objects
#     train_batch = Batch.from_data_list(train_data_list) if train_data_list else None
#     test_batch = Batch.from_data_list(test_data_list) if test_data_list else None
#
#     return train_batch, test_batch


# def create_batch_from_patients_events(patients_events_df, train_patiens, test_patiens):
#     data_list = []
#     for pt_index in patients_events_df['pt_index'].unique():
#         events = patients_events_df[patients_events_df['pt_index'] == pt_index]
#         if not events.empty:
#             data = create_graph_data_from_events(events)
#             data_list.append(data)
#
#     batch_data = Batch.from_data_list(data_list)
#     return batch_data


# Now define a GCN that outputs graph-level embeddings for each patient.
class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch  # data.batch indicates graph membership
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        # Global pooling: one embedding per graph (i.e. per patient)
        x = global_mean_pool(x, batch)
        x = self.lin(x)
        return x

class GAT(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, out_channels, heads=1):
        super(GAT, self).__init__()
        # First GAT layer: input -> hidden
        self.conv1 = GATConv(num_features, hidden_channels, heads=heads)
        # Second GAT layer: (hidden * heads) -> hidden
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=heads)
        # Linear layer: (hidden * heads) -> output
        self.lin = torch.nn.Linear(hidden_channels * heads, out_channels)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        x = global_mean_pool(x, batch)
        x = self.lin(x)

        return x

def generate_graph(crf_df, patients_data, D_0_TIME):
    patients_events = create_events(crf_df, patients_data, D_0_TIME)
    patients_events_df = convert_to_dataframe(patients_events)
    # patients_events_df = patients_events_df.dropna()
    patients_events_df = patients_events_df.dropna().reset_index(drop=True)  # **CHANGE THIS LINE**

    scaler = MinMaxScaler()  # Initialize the scaler
    numeric_cols = [col for col in patients_events_df.columns if
                    col != 'pt_index']  # Select numeric columns for scaling
    patients_events_df[numeric_cols] = scaler.fit_transform(patients_events_df[numeric_cols])  # Apply scaling
    batch_data = create_batch_from_patients_events(patients_events_df)

    # check condition
    print("patients_events_df shape:", patients_events_df.shape)
    print("Unique pt_index values:", patients_events_df['pt_index'].unique())
    print("Max pt_index value:", patients_events_df['pt_index'].max())

    for data in batch_data.to_data_list():
        # print(f"Graph Data - x.shape: {data.x.shape}, edge_index.shape: {data.edge_index.shape}")

        if data.edge_index.numel() > 0:
            print(f"Max edge index: {data.edge_index.max().item()}, Expected max: {data.x.shape[0] - 1}")

    # Model parameters
    num_features = 5  # ['bgl', 'pre_bgl', 'insulin', 'calorie', 'duration']
    hidden_channels = 16  # Adjust as needed
    out_channels = 8  # Graph-level embedding dimension

    model = GCN(num_features, hidden_channels, out_channels)

    # Run a forward pass to get graph-level embeddings for each patient.
    model.eval()
    with torch.no_grad():
        graph_embeddings = model(batch_data)

    print("Graph-level embeddings shape:", graph_embeddings.shape)
    return graph_embeddings


# generate_graph(crf_df, patients_data, D_0_TIME)


def create_all_patient_graphs_nx(crf_df, patients_events):
    """
    Creates a dictionary mapping each patient (from crf_df.index) to a directed graph.
    Each graph's nodes represent events (with event attributes) and edges connect events in order.

    Parameters:
      crf_df: DataFrame whose index contains patient identifiers.
      patients_events: List of event lists corresponding to each patient.

    Returns:
      A dictionary where keys are patient identifiers and values are DiGraph objects.
    """
    patient_graphs = {}
    for pt, events in zip(crf_df.index, patients_events):
        G = nx.DiGraph()
        if events:
            for idx, event in enumerate(events):
                node_code = f"event_{idx}"
                G.add_node(node_code, **event)
                if idx > 0:
                    G.add_edge(f"event_{idx - 1}", node_code)
        # If there are no events, G remains an empty graph.
        patient_graphs[pt] = G
    return patient_graphs

def plot_patient_graph_nx(patient_graphs, pt):
    # Get the patient's graph
    G = patient_graphs[pt]

    # Define a layout for the nodes
    pos = nx.spring_layout(G)

    # Draw the graph with node labels
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=2000)

    # Display the graph
    plt.show()


