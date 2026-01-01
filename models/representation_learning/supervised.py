from models.encoder_layers.impala_cnn import ImpalaCNNLarge, ImpalaCNNSmall
from models.encoder_layers.nature_cnn import NatureCNN
from models.representation_learning_utils.supervised import SupervisedLearningHead
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from typing import TYPE_CHECKING
import torch
if TYPE_CHECKING:
    import gymnasium as gym

class SupervisedRepresentationLearner(BaseFeaturesExtractor):

    def __init__(self,
        observation_space: gym.Space,
        obs_encoder: dict,
        representation_vector: dict,
        projection_architecture: list,
        rssm_configs: dict,
        optimizer_params: dict,
        observation_encoder_dim: int = 256,
        expert_obs: gym.Space= None,
        num_actions: int=3,
        normalized_image: bool = False):
        super(SupervisedRepresentationLearner).__init__(
            observation_space, 
            representation_vector.vector_size_per_factor*representation_vector.num_factors
        )
        # define and create visual encoder
        observation_encoders = {
            'impala-large': ImpalaCNNLarge,
            'impala-small': ImpalaCNNSmall,
            'nature': NatureCNN
        }
        self.observation_encoder = observation_encoders[obs_encoder.encoder](
            observation_space=observation_space,
            features_dim=observation_encoder_dim,
            normalized_image=normalized_image
        )

        # if needed define dreamer v2 style RSSM
        if obs_encoder.enable_rssm:
            raise NotImplementedError("to be implemented with DV2 updates")
        
        # define supervised learning head
        self.supervised_learning_head = SupervisedLearningHead(
            observation_encoder_dim = observation_encoder_dim, 
            vector_size_per_factor = representation_vector.vector_size_per_factor,
            num_factors = representation_vector.num_factors, 
            expert_obs = expert_obs,
            projection_architecture = projection_architecture
        )

        # define additional variables for auxiliary objective
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.optimizer_params = optimizer_params
        
    def forward(self, x:torch.Tensor, actions:torch.Tensor=None, test:bool=True)->torch.Tensor:
        x = self.observation_encoder(x)
        x = self.supervised_learning_head(x, test=test)
        return x
    
    def compute_loss(self, batch_dictionary):
        # convert to tensors
        observations = torch.as_tensor(batch_dictionary.observations).float().to(self.device)
        # needs to be 1-hot encoded vectors for the label
        labels = torch.as_tensor(batch_dictionary.labels).to(self.device)
        
        # forward pass
        with torch.set_grad_enabled(True):
            pred_features = self.forward(observations, test=False)
            loss = 0
            accuracy = 0

            for i, feat in enumerate(pred_features):
                loss += self.loss_fn(feat, labels[:,i])
                # get predicted class indices
                preds = torch.argmax(feat, dim=1)
                accuracy += (preds == labels[:,i]).float().mean()
            loss /= (len(pred_features))
            accuracy /= (len(pred_features))
        return loss, {'accuracy': (accuracy, False)}

    def build_optimizers(self):
        optim_kwargs = {
            "lr": self.optimizer_params.get("lr",1e-4),
            "betas": tuple(self.optimizer_params.get("betas", (0.9, 0.999))),
            "weight_decay": self.optimizer_params.get("weight_decay", 0.0),
            "eps": self.optimizer_params.get("eps", 1e-8),
        }
        return { 
            'all': torch.optim.Adam(
                self.parameters(),
                **optim_kwargs
            )
        }

    def post_step(self):
        return {}