__author__ = 'Shayhan'
import numpy as np
import copy, random, torch, pickle
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from collections import deque
import os
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from graph_generation import *

def reward_calculation(y, desired_bst = 130):
    # return -abs(y - desired_bst)
    # return np.where((y >= 80) & (y <= 180), 1, 0)
    return -(10 * (3.5506 * (np.log(y) ** 0.8353 - 3.7932)) ** 2)


class patient_GCN(torch.nn.Module):
    # def __init__(self, num_features, hidden_channels, out_channels, ax_features=5):
    def __init__(self, graph_in_dim, graph_hidden_dim, graph_out_dim, ax_in_dim, ax_out_dim, out_channels):
        super(patient_GCN, self).__init__()
        self.conv1 = GCNConv(graph_in_dim, graph_hidden_dim)
        self.conv2 = GCNConv(graph_hidden_dim, graph_hidden_dim)
        self.lin1 = torch.nn.Linear(graph_hidden_dim, graph_out_dim)
        self.ax_layer = torch.nn.Linear(ax_in_dim, ax_out_dim)
        self.lin2 = torch.nn.Linear(graph_out_dim+ax_out_dim, out_channels)

    def forward(self, data, ax_data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = global_mean_pool(x, batch)  # Graph-level representation
        x = self.lin1(x)  # Graph embeddings
        ax_x = self.ax_layer(ax_data)
        x = torch.cat((x, ax_x), dim=1)  # Concatenate graph embeddings with axial data
        x = self.lin2(x)
        return x

class patient_GAT(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, out_channels, heads=1):
        super(patient_GAT, self).__init__()
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

        # Graph-level pooling
        x = global_mean_pool(x, batch)

        return self.lin(x)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)

        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.sigmoid(self.l3(a))
        # return self.max_action * torch.tanh(self.l3(a))
        # return 0.5 * self.max_action * (torch.tanh(self.l3(a)) + 1)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)

        # Q-value prediction head
        self.q_head = nn.Linear(256, 1)
        # Auxiliary task prediction head
        # self.aux_head = nn.Linear(256, 1)


    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        x = F.relu(self.l1(sa))
        x = F.relu(self.l2(x))

        # Outputs
        q1 = self.q_head(x)
        # aux_out = self.aux_head(x)

        return q1


class td3_bc:
    # def __init__(self, max_action, state_size=2, action_size=1):
    # def __init__(self, max_action, num_rl_in_dim, num_graph_in_dim, num_graph_out_dim, all_patient_graph_list, hidden_dim=16, action_size=1):
    def __init__(self, max_action, rl_in_dim, graph_in_dim, graph_hidden_dim, graph_out_dim, ax_in_dim, ax_out_dim, pf_out_dim, all_patient_graph_list, action_size=1):
        self.all_patient_graph_list = all_patient_graph_list
        self.num_rl_features = rl_in_dim
        self.graph_in_dim = graph_in_dim
        self.graph_hidden_dim = graph_hidden_dim
        self.graph_out_dim = graph_out_dim
        self.ax_in_dim = ax_in_dim
        self.ax_out_dim = ax_out_dim
        self.pf_out_dim = pf_out_dim  # personalized features
        self.state_size = rl_in_dim + pf_out_dim  # Combined state size
        self.action_size, self.max_action = action_size, max_action
        self.device = device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gcn_lr = 3e-4
        self.actor_lr = 3e-4
        self.critic_lr = 3e-4
        # self.alpha = 2.5
        self.alpha = 100 #org
        self.beta = 0.9 #org

        # self.alpha = 100
        # self.beta = 100


    def init_model(self):
        # Initialize model
        self.gcn = patient_GCN(graph_in_dim=self.graph_in_dim, graph_hidden_dim=self.graph_hidden_dim,
                               graph_out_dim=self.graph_out_dim, ax_in_dim=self.ax_in_dim, ax_out_dim=self.ax_out_dim,
                               out_channels=self.pf_out_dim).to(self.device)
        self.actor = Actor(self.state_size, self.action_size, self.max_action).to(self.device)
        # self.actor_target = copy.deepcopy(self.actor)
        self.critic = Critic(self.state_size, self.action_size).to(self.device)
        # self.critic_target = copy.deepcopy(self.critic)

        # Optimizers
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)
        self.gcn_optimizer = torch.optim.Adam(self.gcn.parameters(), lr=self.gcn_lr)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        # self.actor_gcn_optimizer = torch.optim.Adam(
        #     list(self.actor.parameters()) + list(self.gcn.parameters()), lr=self.actor_lr
        # )


    def select_action(self, dataloader_test, max_action):
        # Feed state into model
        insulin_min = 3.4 # ALL
        for batch in dataloader_test:
            # Move data to device
            # batch_states, batch_actions, batch_rewards = [x.to(self.device) for x in batch]

            b_states, batch_actions, batch_rewards, batch_flt_pts, batch_x_test, batch_ax_data = [x.to(self.device) for x in batch]
            batch_flt_pts = batch_flt_pts.squeeze(1)
            batch_graph_data = create_batch_from_patients_events(self.all_patient_graph_list,
                                                                 batch_flt_pts.cpu().numpy().tolist())
            batch_graph_data = batch_graph_data.to(self.device)
            # Pass graph data through GCN to get embeddings
            graph_embeddings = self.gcn(batch_graph_data, batch_ax_data)
            batch_states = torch.cat((b_states, graph_embeddings), dim=1)

            with torch.no_grad():
                # tensor_state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
                tensor_action = self.actor(batch_states)
                # tensor_action[batch_x_test < 80] -= insulin_min
                tensor_action[batch_x_test <= 80] -= insulin_min  # ALL
                # insulin_min = 2.27  # MIMIC
                # tensor_action[batch_x_test <= 80] -= insulin_min  # MIMIC
                # tensor_action[(batch_x_test > 80) & (batch_x_test <= 130)] -= 2.9  # MIMIC
                # tensor_action[batch_x_test > 130] -= 2  # MIMIC
                # torch.clamp(tensor_action, 0, max_action)
                tensor_action = torch.clamp(tensor_action, 0, max_action)
                # print(tensor_action[batch_x_test < 80])
                # print(f"{batch_actions.shape=}, {tensor_action.shape=}")
                #
                # # Calculate BGL loss with original batch_actions
                # _, org_bgl_pred= self.critic(batch_states, batch_actions)
                # # bgl_loss = F.mse_loss(aux_predictions, batch_y)
                #
                # # Calculate INS loss with actions predicted by the actor
                # _, ins_bgl_pred = self.critic(batch_states, tensor_action)
                # # ins_loss = F.mse_loss(aux_predictions, batch_y)


        return tensor_action.cpu().data.numpy().flatten()
        # return tensor_action.cpu().data.numpy().flatten(), batch_y.cpu().numpy().flatten(), org_bgl_pred.cpu().numpy().flatten(), ins_bgl_pred.cpu().numpy().flatten()

    def train_model(self, train_dataset, max_epochs):
        # Initialize the networks
        self.init_model()
        beta = 0.1
        batch_size = 64
        dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(max_epochs):
            total_critic_loss = 0.0  # Accumulate critic loss
            total_actor_loss = 0.0  # Accumulate actor loss
            num_batches = 0  # Count batches

            # Process data in batches
            for batch in dataloader:
                # Move data to device
                b_states, batch_actions, batch_rewards, batch_flt_pts, batch_y, batch_ax_data = [x.to(self.device) for x in batch]
                batch_flt_pts = batch_flt_pts.squeeze(1)
                batch_graph_data = create_batch_from_patients_events(self.all_patient_graph_list, batch_flt_pts.cpu().numpy().tolist())
                batch_graph_data = batch_graph_data.to(self.device)
                # Pass graph data through GCN to get embeddings
                graph_embeddings = self.gcn(batch_graph_data, batch_ax_data)
                batch_states = torch.cat((b_states, graph_embeddings), dim=1)
                # Update the critic
                # target_Q = batch_rewards.unsqueeze(1)  # Add a new dimension to target_Q for compatibility with critic's output shape'
                target_Q = batch_rewards
                # === Critic pass ===
                current_Q = self.critic(batch_states, batch_actions)
                critic_loss = F.mse_loss(current_Q, target_Q)
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

                # === GCN pass With Critic loss ===
                # re-run the forward pass (fresh graph)
                graph_embeddings = self.gcn(batch_graph_data, batch_ax_data)
                batch_states = torch.cat((b_states, graph_embeddings), dim=1)
                current_Q = self.critic(batch_states, batch_actions)  # or freeze critic
                gcn_loss = F.mse_loss(current_Q, target_Q)
                self.gcn_optimizer.zero_grad()
                gcn_loss.backward()
                self.gcn_optimizer.step()

                num_batches += 1

            # Delayed policy updates
            for batch in dataloader:
                # Move data to device
                # batch_states, batch_actions, batch_rewards = [x.to(self.device) for x in batch]
                b_states, batch_actions, batch_rewards, batch_flt_pts, batch_y, batch_ax_data = [x.to(self.device) for x in batch]
                batch_flt_pts = batch_flt_pts.squeeze(1)
                batch_graph_data = create_batch_from_patients_events(self.all_patient_graph_list, batch_flt_pts.cpu().numpy().tolist())
                batch_graph_data = batch_graph_data.to(self.device)
                # Pass graph data through GCN to get embeddings
                graph_embeddings = self.gcn(batch_graph_data, batch_ax_data)
                batch_states = torch.cat((b_states, graph_embeddings), dim=1)

                # Compute the modified actor loss
                predictions = self.actor(batch_states)
                Q = self.critic(batch_states, predictions)
                lmbda = self.alpha / Q.abs().mean().detach()

                actor_loss = -self.alpha * Q.mean() + self.beta*F.mse_loss(predictions, batch_actions)
                total_actor_loss += actor_loss.item()

                # === Actor pass ===`

                self.actor_optimizer.zero_grad()
                actor_loss.backward(retain_graph=True)  # Keep graph for GCN
                self.actor_optimizer.step()

                # === GCN pass With Post-EN BGL prediction ===`
                # graph_embeddings = self.gcn(batch_graph_data)
                # gcn_loss =F.mse_loss(graph_embeddings, batch_y)
                # self.gcn_optimizer.zero_grad()
                # gcn_loss.backward()  # Backpropagate gradients for GCN
                # self.gcn_optimizer.step()

    # def train_model(self, train_dataset, max_epochs):
    #     # Initialize the networks
    #     self.init_model()
    #     beta = 0.1
    #     batch_size = 64
    #     dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    #
    #     for epoch in range(max_epochs):
    #         total_critic_loss = 0.0  # Accumulate critic loss
    #         total_actor_loss = 0.0  # Accumulate actor loss
    #         num_batches = 0  # Count batches
    #
    #         # Process data in batches
    #         for batch in dataloader:
    #             # Move data to device
    #             b_states, batch_actions, batch_rewards, batch_flt_pts, batch_y = [x.to(self.device) for x in batch]
    #             batch_flt_pts = batch_flt_pts.squeeze(1)
    #             batch_graph_data = create_batch_from_patients_events(self.all_patient_graph_list,
    #                                                                  batch_flt_pts.cpu().numpy().tolist())
    #             batch_graph_data = batch_graph_data.to(self.device)
    #             # Pass graph data through GCN to get embeddings
    #             graph_embeddings = self.gcn(batch_graph_data)
    #             batch_states = torch.cat((b_states, graph_embeddings), dim=1)
    #
    #             target_Q = batch_rewards
    #             # === Critic pass ===
    #             current_Q = self.critic(batch_states, batch_actions)
    #             critic_loss = F.mse_loss(current_Q, target_Q)
    #             self.critic_optimizer.zero_grad()
    #             critic_loss.backward()
    #             self.critic_optimizer.step()
    #
    #             # === GCN pass ===
    #             # re-run the forward pass (fresh graph)
    #             graph_embeddings = self.gcn(batch_graph_data)
    #             batch_states = torch.cat((b_states, graph_embeddings), dim=1)
    #             current_Q = self.critic(batch_states, batch_actions)  # or freeze critic
    #             gcn_loss = F.mse_loss(current_Q, target_Q)
    #             self.gcn_optimizer.zero_grad()
    #             gcn_loss.backward()
    #             self.gcn_optimizer.step()
    #
    #             num_batches += 1
    #
    #         # Delayed policy updates
    #         for batch in dataloader:
    #             # Move data to device
    #             # batch_states, batch_actions, batch_rewards = [x.to(self.device) for x in batch]
    #             b_states, batch_actions, batch_rewards, batch_flt_pts, batch_y = [x.to(self.device) for x in batch]
    #             batch_flt_pts = batch_flt_pts.squeeze(1)
    #             batch_graph_data = create_batch_from_patients_events(self.all_patient_graph_list,
    #                                                                  batch_flt_pts.cpu().numpy().tolist())
    #             batch_graph_data = batch_graph_data.to(self.device)
    #             # Pass graph data through GCN to get embeddings
    #             graph_embeddings = self.gcn(batch_graph_data)
    #             batch_states = torch.cat((b_states, graph_embeddings), dim=1)
    #
    #             # Compute the modified actor loss
    #             predictions = self.actor(batch_states)
    #             Q = self.critic(batch_states, predictions)
    #             lmbda = self.alpha / Q.abs().mean().detach()
    #
    #             actor_loss = -self.alpha * Q.mean() + self.beta * F.mse_loss(predictions, batch_actions)
    #             total_actor_loss += actor_loss.item()
    #
    #             self.actor_optimizer.zero_grad()
    #             actor_loss.backward(retain_graph=True)  # Keep graph for GCN
    #             self.actor_optimizer.step()

    # def train_model(self, train_dataset, max_epochs):
    #     # Initialize the networks
    #     self.init_model()
    #     beta = 0.1
    #
    #     # Create a dataset and DataLoader
    #     # dataset = TensorDataset(train_dataset)
    #     batch_size = 64
    #     dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    #
    #     for epoch in range(max_epochs):
    #         total_critic_loss = 0.0  # Accumulate critic loss
    #         # total_aux_loss = 0.0  # Accumulate auxiliary loss
    #         total_actor_loss = 0.0  # Accumulate actor loss
    #         num_batches = 0  # Count batches
    #
    #         # Process data in batches
    #         for batch in dataloader:
    #             # Move data to device
    #             b_states, batch_actions, batch_rewards, batch_flt_pts = [x.to(self.device) for x in batch]
    #             batch_flt_pts = batch_flt_pts.squeeze(1)
    #             batch_graph_data = create_batch_from_patients_events(self.all_patient_graph_list,
    #                                                                  batch_flt_pts.cpu().numpy().tolist())
    #             batch_graph_data = batch_graph_data.to(self.device)
    #             # Pass graph data through GCN to get embeddings
    #             graph_embeddings = self.gcn(batch_graph_data)
    #             batch_states = torch.cat((b_states, graph_embeddings), dim=1)
    #
    #             # Update the critic
    #             target_Q = batch_rewards.unsqueeze(1)  # Add a new dimension to target_Q for compatibility with critic's output shape'
    #             # print(f"{batch_states.shape=}")
    #             current_Q = self.critic(batch_states, batch_actions)
    #             critic_loss = F.mse_loss(current_Q, target_Q)
    #             # print(f"Train {batch_actions.shape=}")
    #
    #
    #             # # Compute losses
    #             # theta = 0.1
    #             # critic_loss = F.mse_loss(current_Q1, target_Q)
    #             # # print(f"{current_Q1.shape=}, {target_Q.shape=}")
    #             # # print(f"{aux_predictions.shape=}, {batch_y.shape=}")
    #             # aux_loss = F.mse_loss(aux_predictions, batch_y)
    #             # total_loss = critic_loss + theta * aux_loss  # Combine losses with weight beta
    #
    #             # Backpropagate
    #             self.critic_optimizer.zero_grad()
    #             critic_loss.backward()
    #             self.critic_optimizer.step()
    #
    #             # Accumulate losses for logging
    #             # total_critic_loss += critic_loss.item()
    #             # total_aux_loss += aux_loss.item()
    #             num_batches += 1
    #
    #         # Delayed policy updates
    #         for batch in dataloader:
    #             # Move data to device
    #             # batch_states, batch_actions, batch_rewards = [x.to(self.device) for x in batch]
    #             b_states, batch_actions, batch_rewards, batch_flt_pts = [x.to(self.device) for x in batch]
    #             batch_flt_pts = batch_flt_pts.squeeze(1)
    #             batch_graph_data = create_batch_from_patients_events(self.all_patient_graph_list,
    #                                                                  batch_flt_pts.cpu().numpy().tolist())
    #             batch_graph_data = batch_graph_data.to(self.device)
    #             # Pass graph data through GCN to get embeddings
    #             graph_embeddings = self.gcn(batch_graph_data)
    #             batch_states = torch.cat((b_states, graph_embeddings), dim=1)
    #
    #             # Compute the modified actor loss
    #             predictions = self.actor(batch_states)
    #             Q = self.critic(batch_states, predictions)
    #             lmbda = self.alpha / Q.abs().mean().detach()
    #
    #             actor_loss = -self.alpha * Q.mean() + self.beta*F.mse_loss(predictions, batch_actions)
    #             # Compute GCN Loss (Use the same `actor_loss` for gradient flow)
    #             gcn_loss = actor_loss  # Ensures gradients flow through GCN
    #
    #             # Compute total loss (for joint optimization)
    #             total_loss = actor_loss + gcn_loss  # Adjust weight if necessary
    #
    #             # Backpropagate once for both Actor and GCN
    #             total_loss.backward()
    #             self.actor_gcn_optimizer.step()  # Update both Actor and GCN at the same time



    # def train_model(self, train_dataset, max_epochs):
    #     # Initialize the networks
    #     self.init_model()
    #     beta = 0.1
    #
    #     # Create a dataset and DataLoader
    #     # dataset = TensorDataset(train_dataset)
    #     batch_size = 64
    #     dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    #
    #     for epoch in range(max_epochs):
    #         total_critic_loss = 0.0  # Accumulate critic loss
    #         # total_aux_loss = 0.0  # Accumulate auxiliary loss
    #         total_actor_loss = 0.0  # Accumulate actor loss
    #         num_batches = 0  # Count batches
    #
    #         # Process data in batches
    #         for batch in dataloader:
    #             # Move data to device
    #             b_states, batch_actions, batch_rewards, batch_flt_pts, batch_y = [x.to(self.device) for x in batch]
    #             batch_flt_pts = batch_flt_pts.squeeze(1)
    #             batch_graph_data = create_batch_from_patients_events(self.all_patient_graph_list,
    #                                                                  batch_flt_pts.cpu().numpy().tolist())
    #             batch_graph_data = batch_graph_data.to(self.device)
    #             # Pass graph data through GCN to get embeddings
    #             graph_embeddings = self.gcn(batch_graph_data)
    #             batch_states = torch.cat((b_states, graph_embeddings), dim=1)
    #
    #             # Update the critic
    #             # target_Q = batch_rewards.unsqueeze(1)  # Add a new dimension to target_Q for compatibility with critic's output shape'
    #             target_Q = batch_rewards
    #             # print(f"{batch_states.shape=}")
    #             current_Q = self.critic(batch_states, batch_actions)
    #             critic_loss = F.mse_loss(current_Q, target_Q)
    #             # print(f"Train {batch_actions.shape=}")
    #
    #
    #             # # Compute losses
    #             # theta = 0.1
    #             # critic_loss = F.mse_loss(current_Q1, target_Q)
    #             # # print(f"{current_Q1.shape=}, {target_Q.shape=}")
    #             # # print(f"{aux_predictions.shape=}, {batch_y.shape=}")
    #             # aux_loss = F.mse_loss(aux_predictions, batch_y)
    #             # total_loss = critic_loss + theta * aux_loss  # Combine losses with weight beta
    #
    #             # Backpropagate
    #             self.critic_optimizer.zero_grad()
    #             critic_loss.backward()
    #             self.critic_optimizer.step()
    #
    #             # Accumulate losses for logging
    #             # total_critic_loss += critic_loss.item()
    #             # total_aux_loss += aux_loss.item()
    #             num_batches += 1
    #
    #         # Delayed policy updates
    #         for batch in dataloader:
    #             # Move data to device
    #             # batch_states, batch_actions, batch_rewards = [x.to(self.device) for x in batch]
    #             b_states, batch_actions, batch_rewards, batch_flt_pts, batch_y = [x.to(self.device) for x in batch]
    #             batch_flt_pts = batch_flt_pts.squeeze(1)
    #             batch_graph_data = create_batch_from_patients_events(self.all_patient_graph_list,
    #                                                                  batch_flt_pts.cpu().numpy().tolist())
    #             batch_graph_data = batch_graph_data.to(self.device)
    #             # Pass graph data through GCN to get embeddings
    #             graph_embeddings = self.gcn(batch_graph_data)
    #             batch_states = torch.cat((b_states, graph_embeddings), dim=1)
    #
    #             # Compute the modified actor loss
    #             predictions = self.actor(batch_states)
    #             Q = self.critic(batch_states, predictions)
    #             lmbda = self.alpha / Q.abs().mean().detach()
    #
    #             actor_loss = -self.alpha * Q.mean() + self.beta*F.mse_loss(predictions, batch_actions)
    #             total_actor_loss += actor_loss.item()
    #
    #
    #             # self.actor_optimizer.zero_grad()
    #             # actor_loss.backward()
    #             # self.actor_optimizer.step()
    #
    #             self.actor_optimizer.zero_grad()
    #             actor_loss.backward(retain_graph=True)  # Keep graph for GCN
    #             self.actor_optimizer.step()
    #
    #
    #             #-----
    #             # Compute x and y coordinates
    #
    #             predictions = self.actor(batch_states)
    #             # Ensure tensors require gradients
    #             batch_y = batch_y.requires_grad_(True)
    #             predictions = predictions.requires_grad_(True)
    #             batch_actions = batch_actions.requires_grad_(True)
    #
    #             # Compute GCN loss with smooth differentiable quadrants
    #             x_axis = batch_y - 130
    #             y_axis = predictions - batch_actions
    #
    #             # q1 = torch.sigmoid(10 * (x_axis)) * torch.sigmoid(10 * (y_axis))
    #             # q2 = (1 - torch.sigmoid(10 * (x_axis))) * torch.sigmoid(10 * (y_axis))
    #             # q3 = (1 - torch.sigmoid(10 * (x_axis))) * (1 - torch.sigmoid(10 * (y_axis)))
    #             # q4 = torch.sigmoid(10 * (x_axis)) * (1 - torch.sigmoid(10 * (y_axis)))
    #
    #             q1 = torch.sigmoid(10 * (x_axis)) * torch.sigmoid(10 * (y_axis))
    #             q2 = (1 - torch.sigmoid(10 * (x_axis))) * torch.sigmoid(10 * (y_axis))
    #             q3 = (1 - torch.sigmoid(10 * (x_axis))) * (1 - torch.sigmoid(10 * (y_axis)))
    #             q4 = torch.sigmoid(10 * (x_axis)) * (1 - torch.sigmoid(10 * (y_axis)))
    #
    #             total = batch_y.numel()
    #             gcn_loss = (torch.sum(q2 + q4) - torch.sum(q1 + q3)) / total  # Differentiable loss
    #
    #             # ---- Update GCN ----
    #             self.gcn_optimizer.zero_grad()
    #             gcn_loss.backward()  # Backpropagate gradients for GCN
    #             self.gcn_optimizer.step()





                # # Recompute `gcn_loss` instead of using `actor_loss.clone()`
                # graph_embeddings = self.gcn(batch_graph_data)
                # batch_states = torch.cat((b_states, graph_embeddings), dim=1)
                #
                # # Compute a new loss for GCN, ensuring gradients flow properly
                # predictions = self.actor(batch_states)
                # Q = self.critic(batch_states, predictions)
                # gcn_loss = -self.alpha * Q.mean() + self.beta * F.mse_loss(predictions, batch_actions)

                # ### ---- Update GCN ---- ###
                # self.gcn_optimizer.zero_grad()
                # gcn_loss.backward()  # Backpropagate gradients for GCN
                # self.gcn_optimizer.step()




    # def train_model(self, dataloader, graph_data, max_epochs=25):
    #     # Initialize the networks
    #     self.init_model()
    #     beta = 0.1
    #
    #     # Create a dataset and DataLoader
    #     # dataset = TensorDataset(train_dataset)
    #     batch_size = 64
    #     dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    #
    #     for epoch in range(max_epochs):
    #         total_critic_loss = 0.0  # Accumulate critic loss
    #         # total_aux_loss = 0.0  # Accumulate auxiliary loss
    #         total_actor_loss = 0.0  # Accumulate actor loss
    #         num_batches = 0  # Count batches
    #
    #         # Process data in batches
    #         for batch in dataloader:
    #             # Move data to device
    #             b_states, batch_actions, batch_rewards, batch_flt_pts = [x.to(self.device) for x in batch]
    #             batch_flt_pts = batch_flt_pts.squeeze(1)
    #             batch_graph_data = create_batch_from_patients_events(self.all_patient_graph_list,
    #                                                                  batch_flt_pts.cpu().numpy().tolist())
    #             batch_graph_data = batch_graph_data.to(self.device)
    #             # Pass graph data through GCN to get embeddings
    #             graph_embeddings = self.gcn(batch_graph_data)
    #             batch_states = torch.cat((b_states, graph_embeddings), dim=1)
    #
    #             # Update the critic
    #             target_Q = batch_rewards.unsqueeze(1)  # Add a new dimension to target_Q for compatibility with critic's output shape'
    #             # print(f"{batch_states.shape=}")
    #             current_Q = self.critic(batch_states, batch_actions)
    #             critic_loss = F.mse_loss(current_Q, target_Q)
    #             # print(f"Train {batch_actions.shape=}")
    #
    #             # Backpropagate
    #             self.critic_optimizer.zero_grad()
    #             critic_loss.backward()
    #             self.critic_optimizer.step()
    #
    #             # Compute the modified actor loss
    #             predictions = self.actor(batch_states)
    #             Q = self.critic(batch_states, predictions)
    #             lmbda = self.alpha / Q.abs().mean().detach()
    #
    #             actor_loss = -self.alpha * Q.mean() + self.beta * F.mse_loss(predictions, batch_actions)
    #             # Compute GCN Loss (Use the same `actor_loss` for gradient flow)
    #             gcn_loss = actor_loss  # Ensures gradients flow through GCN
    #
    #             # Compute total loss (for joint optimization)
    #             total_loss = actor_loss + gcn_loss  # Adjust weight if necessary
    #
    #             # Backpropagate once for both Actor and GCN
    #             total_loss.backward()
    #             self.actor_gcn_optimizer.step()  # Update both Actor and GCN at the same time