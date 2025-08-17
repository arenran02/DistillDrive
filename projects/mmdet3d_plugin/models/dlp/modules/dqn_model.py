import torch
import torch.nn as nn
from mmdet.models import build_loss
from mmcv.cnn.bricks.registry import PLUGIN_LAYERS

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 24) 
        self.fc2 = nn.Linear(24, 24) 
        self.fc3 = nn.Linear(24, action_size)  

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        q_values = self.fc3(x) 
        return q_values

@PLUGIN_LAYERS.register_module()
class DQNAgent(nn.Module):
    def __init__(
        self,
        state_size=10,
        action_size=3,
        gamma=0.95,
        dqn_loss=None
        ):
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma

        self.model = DQN(state_size, action_size)
        self.target_model = DQN(state_size, action_size)
        self.dqn_loss = build_loss(dqn_loss)
        self.reward_weight = torch.nn.Parameter(torch.randn(6))


    def reward_func(
        self,
        gt_status,
        pred_status,
        reg_target,
        reg_pred,        
    ):
        '''
            gt_status: [B, 10],
            pred_status: [B, 10],
            reg_target: [B, T, 2],
            reg_pred: [B, T, 2]
        '''
        status_erro = torch.mean(torch.abs(gt_status - pred_status), dim=1)
        status_reward = torch.exp(-status_erro).detach()

        pos_pred_by_speed = pred_status[..., 6] * 0.5
        consist_erro = torch.abs(reg_pred[:, 0,1] - pos_pred_by_speed)
        consist_reward = torch.exp(-consist_erro).detach()

        speed_limit_reward = ((pred_status[..., 6] > 0 ) & (pred_status[..., 6] < 20)).float()

        reg_erro = torch.mean(torch.abs(reg_target - reg_pred), dim=(1,2))
        reg_reward = torch.exp(-reg_erro).detach()

        start_erro = torch.mean(torch.abs(reg_target[:, 0,:] - reg_pred[:, 0,:]), dim=(1))
        start_reward = torch.exp(-start_erro).detach()

        end_erro = torch.mean(torch.abs(reg_target.cumsum(dim=1)[:, -1,:] - reg_pred.cumsum(dim=1)[:, -1,:]), dim=(1))
        end_reward = torch.exp(-end_erro).detach()

        # Inverse Reinforce Learing
        reward = torch.stack((status_reward, consist_reward, speed_limit_reward, reg_reward, start_reward, end_reward) ,dim=1)
        reward = torch.matmul(reward, self.reward_weight.unsqueeze(1)).squeeze()

        return reward

    def forward(
        self,
        gt_status,
        pred_status,
        reg_target,
        reg_pred,
        ego_cmd
        ):
        # index gather
        bs = gt_status.shape[0]
        bs_indices = torch.arange(bs, device=reg_pred.device)
        cmd = ego_cmd.argmax(dim=-1)
        ego_mask = ~torch.all(gt_status[:,:] == 0, dim=1)
        
        # reward function
        # mat_coef = torch.corrcoef(torch.cat([pred_status, gt_status], dim=0)).detach()
        # reward = mat_coef[bs:, :bs].diagonal()
        reward = self.reward_func(gt_status, pred_status, reg_target, reg_pred)


        gt_action = self.target_model(gt_status)
        target = reward + self.gamma * torch.max(gt_action, dim=1)[0] 
        gt_action_loss = self.dqn_loss(gt_action[ego_mask], ego_cmd[ego_mask])
        
        gt_action[bs_indices, cmd] = target
        pred_action = self.model(pred_status)[bs_indices, cmd]
        pred_action_loss = self.dqn_loss(pred_action[ego_mask], target[ego_mask])
                                                             
        return pred_action_loss + gt_action_loss