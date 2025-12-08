import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
from typing import Optional
# from ema_pytorch import EMA as OtherEMA

class EMA:
      def __init__(self, model, decay):
          """
          Initializes the EMA class.

          Args:
              model (torch.nn.Module): The model to apply EMA to.
              decay (float): The decay factor for EMA.
          """
          self.model = model
          self.decay = decay
          self.ema_model = copy.deepcopy(model)
          for param in self.ema_model.parameters():
              param.requires_grad_(False)

      def update(self):
          """
          Updates the EMA model parameters.
          """
          with torch.no_grad():
              for ema_param, param in zip(self.ema_model.parameters(), self.model.parameters()):
                  ema_param.data.mul_(self.decay).add_(param.data, alpha=1 - self.decay)

class SelfSupervisedCovIKLearner(nn.Module):

    def __init__(self, backbone_dim:int, vector_size_per_factor:int, num_factors:int, num_actions:int, smoothness_margin:float=0.2):
        super(SelfSupervisedCovIKLearner, self).__init__()
        self.factor_scale_proj = nn.Sequential(
                                nn.Linear(backbone_dim, num_factors*vector_size_per_factor),
                                nn.ReLU(),
                                nn.Linear(num_factors*vector_size_per_factor, num_factors*vector_size_per_factor)
        )

        self.act = torch.relu

        self.ik_proj = nn.ModuleList([
                nn.Linear(2 * vector_size_per_factor, num_actions)
                for _ in range(num_factors)
            ])

        #EMA to slow learning for transition projector and down projector
        # self.transition_proj_ema = EMA(self.transition_proj, beta=0.999)
        # self.down_proj_ema = EMA(self.down_proj, beta=0.999)

        #extra variables to reshape output or set margins
        self.num_factors = num_factors
        self.vector_size_per_factor = vector_size_per_factor
        self.smoothness_margin = smoothness_margin
        self.softmax = nn.Softmax(dim=-1)
    

    def forward(self, x: torch.Tensor, actions:Optional[torch.Tensor] = None, test:bool=True)->torch.Tensor:

        #if in eval mode: used for PPO policy learning => then detach computation and return representation as is 
        if test:

            x = self.act(self.factor_scale_proj(x)).detach()

            return x

        
        #if in train mode: used for SSL representation learning ==> then do not detach computation and return both covariance and next state prediction
        else:
            x_proj = self.act(self.factor_scale_proj(x)) #should be: (batch size, num factors*vector_size_per_factor)

            # Compute the mean of each feature
            factor_means = torch.mean(x_proj, dim=0, keepdim=True)

            # Center the representation by subtracting the mean
            x_centered = x_proj - factor_means

            # (1) Compute covariance matrix for batch
            batch_cov = (x_centered.T @ x_centered) / (x_centered.shape[0] - 1)

            # (2) Compare x_{t} with x_{t+1} for action (inverse kinematics) prediction
            x_proj = x_proj.reshape(x_proj.shape[0], self.num_factors, self.vector_size_per_factor)
            pred_action = []
            for i, f in enumerate(range(self.num_factors)):
                ik_input = torch.cat([x_proj[:-1,f,:], x_proj[1:,f,:]], dim=-1) #shape: (batch, 1, 2*vec size per factor)
                pred_action.append(self.softmax(self.ik_proj[i](ik_input)))#shape: [(batch, 1, action_dim)...]
            
            pred_action = torch.stack(pred_action, dim=1) #shape: (batch, factors, action_dim)
        
            #(3) Factor-specific smoothness in predictions
            thresholded_diff = torch.norm(x_proj[:-1,:,:] - x_proj[1:,:,:], p=2, dim=-1) - self.smoothness_margin
            thresholded_diff = self.act(thresholded_diff)
            
            return batch_cov, pred_action, thresholded_diff

class SelfSupervisedCovLearner(nn.Module):

    def __init__(self, backbone_dim: int, vector_size_per_factor:int, num_factors:int, num_actions:int):
        super(SelfSupervisedCovLearner, self).__init__()
        self.factor_scale_proj = nn.Sequential(
                                nn.Linear(backbone_dim, num_factors*vector_size_per_factor),
                                nn.ReLU(),
                                nn.Linear(num_factors*vector_size_per_factor, num_factors*vector_size_per_factor)
        )

        self.act = torch.relu

        self.transition_proj = nn.Linear(num_factors*vector_size_per_factor, num_factors*vector_size_per_factor)
        self.down_proj = nn.Linear(num_factors*vector_size_per_factor, backbone_dim)

        #EMA to slow learning for transition projector and down projector
        # self.transition_proj_ema = EMA(self.transition_proj, beta=0.999)
        # self.down_proj_ema = EMA(self.down_proj, beta=0.999)

        #extra variables to reshape output
        self.num_factors = num_factors
        self.vector_size_per_factor = vector_size_per_factor
    

    def forward(self, x: torch.Tensor, actions:Optional[torch.Tensor] = None, test:bool=True)->torch.Tensor:
        
        
        #if in eval mode: used for PPO policy learning => then detach computation and return representation as is 
        if test:

            x = self.act(self.factor_scale_proj(x)).detach()

            return x

            
        #if in train mode: used for SSL representation learning ==> then do not detach computation and return both covariance and next state prediction
        else:
            x_proj = self.act(self.factor_scale_proj(x)) #should be: (batch size, num factors*vector_size_per_factor)

            # Compute the mean of each feature
            factor_means = torch.mean(x_proj, dim=0, keepdim=True)

            # Center the representation by subtracting the mean
            x_centered = x_proj - factor_means

            # (1) Compute covariance matrix for batch
            batch_cov = (x_centered.T @ x_centered) / (x_proj.shape[0] - 1)

            # (2) Project x_{t} to x_{t+1} (next state estimation)
            #NOTE: ?? do not apply act so that proj remains interpretable (factor-to-factor relation)
            #NOTE: index up to 2nd last index since we don't have reference GT for next state after LAST in batch
            x_hat = self.act(self.transition_proj(x_proj))[:-1] #shape: (batch, num_factors*vector_size_per_factor) #NOTE: use EMA model for next state prediction for slower learning
            
            x_hat = self.down_proj(x_hat) #shape: (batch, backbone_dim) #NOTE: use EMA model for next state prediction for slower learning

            #(3) Return parameters of self.transition_proj for extra loss
            return batch_cov, x, x_hat, self.transition_proj.parameters()


class SelfSupervisedMaskLearner(nn.Module):

    def __init__(self, backbone_dim: int, vector_size_per_factor:int, num_factors:int, num_actions:int):
        super(SelfSupervisedMaskLearner, self).__init__()

        self.factor_scale_proj = nn.Sequential(
                                nn.Linear(backbone_dim, num_factors*vector_size_per_factor),
                                nn.ReLU(),
                                nn.Linear(num_factors*vector_size_per_factor, num_factors*vector_size_per_factor)
        )

        self.act = torch.relu

        # Learnable mask between factors
        self.mask = nn.Parameter(torch.randn(num_factors, num_factors))
        
        # Learnable transformation (2-layer MLP)
        self.transition_proj = nn.Sequential(
                                nn.Linear(num_factors*vector_size_per_factor, num_factors*vector_size_per_factor),
                                nn.ReLU(),
                                nn.Linear(num_factors*vector_size_per_factor, backbone_dim)
        )
        # self.transition_proj_ema = EMA(self.transition_proj, beta=0.999)

        #extra variables to reshape output
        self.num_factors = num_factors
        self.vector_size_per_factor = vector_size_per_factor
    

    def forward(self, x: torch.Tensor,actions:Optional[torch.Tensor] = None, test:bool=True)->torch.Tensor:
        
        #if in eval mode: used for PPO policy learning => then detach computation and return representation as is 
        if test:

            x = self.act(self.factor_scale_proj(x)).detach()
            # x = x.reshape(x.shape[0], self.num_factors, self.vector_size_per_factor).detach() # shape: (batch, num_factors, vec_size)

            return x

            
        #if in train mode: used for SSL representation learning ==> then do not detach computation and return both covariance and next state prediction
        else:
            x_proj = self.act(self.factor_scale_proj(x)) #should be: (batch size, num factors*vec per factor)
            x_curr = x_proj.reshape(x_proj.shape[0], self.num_factors, self.vector_size_per_factor) #should be: (batch size, num factors, vec per factor)

            #(2) Enforce covaraince constraints on mask
            M = torch.sigmoid(self.mask)
            x_curr = torch.einsum('ij,bjv->biv', M, x_curr) # shape: (batch, num_factors, vec_size)
            x_curr = x_curr.reshape(x_curr.shape[0], self.num_factors*self.vector_size_per_factor)
            x_hat = self.act(self.transition_proj(x_curr))[:-1] #NOTE: use EMA model for next state prediction for slower learning

            #(3) Factor-specific smoothness in predictions
            thresholded_diff = torch.norm(x_proj[:-1,:,:] - x_proj[1:,:,:], p=2, dim=-1) - self.smoothness_margin
            thresholded_diff = self.act(thresholded_diff)**2

            return x, x_hat, self.mask, thresholded_diff

class SelfSupervisedMaskReconstrLearner(nn.Module):

    def __init__(self, backbone_dim: int, vector_size_per_factor:int, num_factors:int, num_actions:int):
        super(SelfSupervisedMaskReconstrLearner, self).__init__()

        self.factor_scale_proj = nn.Sequential(
                                nn.Linear(backbone_dim, num_factors*vector_size_per_factor),
                                nn.ReLU(),
                                nn.Linear(num_factors*vector_size_per_factor, num_factors*vector_size_per_factor)
        )

        self.act = torch.relu

        # Learnable mask between factors
        self.mask = nn.Parameter(torch.randn(num_factors, num_factors))
        
        # Learnable transformation (2-layer MLP)
        self.transition_proj = nn.Sequential(
                                nn.Linear(vector_size_per_factor + num_actions, vector_size_per_factor),
                                nn.ReLU()
        )

        #Learnable model to distinguish real from fake transitions
        self.discriminiator = nn.Sequential(
                                nn.Linear(2*vector_size_per_factor, 2)
        )
        self.softmax = nn.Softmax(dim=-1)
        # self.transition_proj_ema = EMA(self.transition_proj, beta=0.999)

        #extra variables to reshape output
        self.num_factors = num_factors
        self.vector_size_per_factor = vector_size_per_factor
        self.num_actions = num_actions
    

    def forward(self, x: torch.Tensor, actions:Optional[torch.Tensor] = None, test:bool=True)->torch.Tensor:
        
        #if in eval mode: used for PPO policy learning => then detach computation and return representation as is 
        if test:
            x = self.act(self.factor_scale_proj(x)).detach()
            # x = x.reshape(x.shape[0], self.num_factors, self.vector_size_per_factor).detach() # shape: (batch, num_factors, vec_size)

            return x

            
        #if in train mode: used for SSL representation learning ==> then do not detach computation and return both covariance and next state prediction
        else:
            x_proj = self.act(self.factor_scale_proj(x)) #should be: (batch size, num factors*vec per factor)
            x_proj = x_proj.reshape(x_proj.shape[0], self.num_factors, self.vector_size_per_factor) #should be: (batch size, num factors, vec per factor)

            #(1) parent factor masking is close to identity and sparse
            M = torch.sigmoid(self.mask)
            x_proj = torch.einsum('ij,bjv->biv', M, x_proj) # shape: (batch, num_factors, vec_size)
            actions = F.one_hot(actions.unsqueeze(1).repeat(1, self.num_factors), num_classes=self.num_actions)          
            x_curr = torch.cat([x_proj[:-1], actions], dim=-1) #shape: (batch, num_factors, vec_size + num_actions)

            #(2) label of 1 for same state and 0 for different state
            x_next = x_proj[1:]
            x_hat = self.transition_proj(x_curr.reshape(x_curr.shape[0]*self.num_factors, self.vector_size_per_factor + self.num_actions))
            # x_hat = x_hat.reshape(-1, self.num_factors, self.vector_size_per_factor) #NOTE: use EMA model for next state prediction for slower learning
            same_state = torch.cat([x_next.reshape(x_next.shape[0]*self.num_factors, self.vector_size_per_factor), x_hat], dim=1)
            same_state = self.softmax(self.discriminiator(same_state))

            perm = torch.randperm(x_next.shape[0], device=x_next.device)
            x_shuffled = x_next.index_select(0, perm).detach() #don't propagate gradients through permuted batch
            diff_state = torch.cat([x_shuffled.reshape(x_shuffled.shape[0]*self.num_factors, self.vector_size_per_factor), x_hat], dim=1)
            diff_state = self.softmax(self.discriminiator(diff_state))

            return M, same_state, diff_state

#TODO: need to implement this
class SelfSupervisedMOE(nn.Module):

    def __init__(self):
        pass