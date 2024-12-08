import argparse
import os
import numpy as np
import torch
import sys
from attrdict import AttrDict

from models import TrajectoryGenerator

parser = argparse.ArgumentParser()
parser.add_argument('--model_path',default='/home/zihanyu/sgan/test_models', type=str)


def get_generator(checkpoint):
    args = AttrDict(checkpoint['args'])
    generator = TrajectoryGenerator(
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        embedding_dim=args.embedding_dim,
        encoder_h_dim=args.encoder_h_dim_g,
        decoder_h_dim=args.decoder_h_dim_g,
        mlp_dim=args.mlp_dim,
        num_layers=args.num_layers,
        noise_dim=args.noise_dim,
        noise_type=args.noise_type,
        noise_mix_type=args.noise_mix_type,
        pooling_type=args.pooling_type,
        pool_every_timestep=args.pool_every_timestep,
        dropout=args.dropout,
        bottleneck_dim=args.bottleneck_dim,
        neighborhood_size=args.neighborhood_size,
        grid_size=args.grid_size,
        batch_norm=args.batch_norm)
    generator.load_state_dict(checkpoint['g_state'])
    # generator.cuda()
    generator.train()
    return generator



class prediction ():
    def __init__(self, model_path = None):
        if model_path:
            self.path = model_path
        else:
            self.path = '/home/zihanyu/sgan/models/sgan-models/eth_12_model.pt' 
    
    def set_history_library(self, history):
        self.history_library = history

    def start_prediction(self, agent_name, data):
        obs_traj = np.array(self.history_library[agent_name])
        obs_traj = torch.from_numpy(obs_traj).type(torch.float)
        obs_traj = obs_traj[:,None,:]
        trajectory_np = np.array(obs_traj).T  # 转置使其按维度分行排列

        relative_trajectory = np.zeros_like(trajectory_np)

        relative_trajectory[:, 1:] = trajectory_np[:, 1:] - trajectory_np[:, :-1]

        obs_traj_rel = relative_trajectory.T
        obs_traj_rel = torch.from_numpy(obs_traj_rel).type(torch.float)
        obs_traj_rel = obs_traj_rel[:,None,:]
        checkpoint = torch.load(self.path,map_location=torch.device('cpu'))
        generator = self.get_generator(checkpoint)
        with torch.no_grad():
            pred_traj_fake_rel = generator(obs_traj, obs_traj_rel, torch.tensor([[0,1]]))
            pred_traj_fake = self.relative_to_abs(pred_traj_fake_rel, obs_traj[-1])
        pred_traj = pred_traj_fake.squeeze(1).numpy()
        pred_x = pred_traj[:10,0]
        pred_y = pred_traj[:10,1]
        return pred_x, pred_y
    
    def get_generator(self, checkpoint):
        args = AttrDict(checkpoint['args'])
        generator = TrajectoryGenerator(
            obs_len=args.obs_len,
            pred_len=args.pred_len,
            embedding_dim=args.embedding_dim,
            encoder_h_dim=args.encoder_h_dim_g,
            decoder_h_dim=args.decoder_h_dim_g,
            mlp_dim=args.mlp_dim,
            num_layers=args.num_layers,
            noise_dim=args.noise_dim,
            noise_type=args.noise_type,
            noise_mix_type=args.noise_mix_type,
            pooling_type=args.pooling_type,
            pool_every_timestep=args.pool_every_timestep,
            dropout=args.dropout,
            bottleneck_dim=args.bottleneck_dim,
            neighborhood_size=args.neighborhood_size,
            grid_size=args.grid_size,
            batch_norm=args.batch_norm)
        generator.load_state_dict(checkpoint['g_state'])
        generator.train()
        return generator
    
    def relative_to_abs(self,rel_traj, start_pos):
        """
        Inputs:
        - rel_traj: pytorch tensor of shape (seq_len, batch, 2)
        - start_pos: pytorch tensor of shape (batch, 2)
        Outputs:
        - abs_traj: pytorch tensor of shape (seq_len, batch, 2)
        """
        # batch, seq_len, 2
        rel_traj = rel_traj.permute(1, 0, 2)
        displacement = torch.cumsum(rel_traj, dim=1)
        start_pos = torch.unsqueeze(start_pos, dim=1)
        abs_traj = displacement + start_pos
        return abs_traj.permute(1, 0, 2)


