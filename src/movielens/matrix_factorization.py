# Description: This file contains the implementation of the Matrix Factorization Model for the FedRecon Model
# Code adapted from https://www.tensorflow.org/federated/tutorials/federated_reconstruction_for_matrix_factorization

import torch
import torch.nn as nn
import torch.functional as F
from typing import List, Tuple, Any
from torchinfo import summary


class ItemEmbedding(nn.Module):
    def __init__(self, num_items, num_latent_factors, spread_out=True):
        super(ItemEmbedding, self).__init__()
        self.num_items = num_items
        self.num_latent_factors = num_latent_factors
        self.tensor_shape = torch.empty(self.num_items, self.num_latent_factors)
        self.item_embedding = nn.Parameter(nn.init.kaiming_uniform_(self.tensor_shape))
        self.spread_out = spread_out
        
    def forward(self, input):
        item_embedding = self.item_embedding[input]    
        return item_embedding

class UserEmbedding(nn.Module):
    def __init__(self, num_latent_factors):
        super(UserEmbedding, self).__init__()
        self.num_latent_factors = num_latent_factors
        self.tensor_shape = torch.empty(1, self.num_latent_factors)
        self.embedding = nn.Parameter(nn.init.kaiming_uniform_(self.tensor_shape))
        
        
    def forward(self, input):
        embedding = self.embedding
        return embedding
    
class FullUserEmbedding(nn.Module):
    def __init__(self, num_users, num_latent_factors):
        super(FullUserEmbedding, self).__init__()
        self.num_users = num_users
        self.num_latent_factors = num_latent_factors
        self.tensor_shape = torch.empty(self.num_users, self.num_latent_factors)
        self.embedding = nn.Parameter(nn.init.kaiming_uniform_(self.tensor_shape))
    
    def forward(self, input):
        embedding = self.embedding[input]
        return embedding
    
    
class AddBias(nn.Module):
    def __init__(self, dim):
        super(AddBias, self).__init__()
        self.tensor_shape = torch.empty(*dim)
        self.bias = nn.Parameter(nn.init.kaiming_uniform_(self.tensor_shape))
        
    def forward(self, ids):
        if ids >= len(self.bias):
            return self.bias
        else:
            return self.bias[ids]


class EmbeddingSpreadoutRegularizer(nn.Module):
    def __init__(self, spreadout_lambda=0.0, l2_normalize=False, l2_regularization=0.0):
        super(EmbeddingSpreadoutRegularizer, self).__init__()
        self.spreadout_lambda = spreadout_lambda
        self.l2_normalize = l2_normalize
        self.l2_regularization = l2_regularization

    def forward(self, weights):
        total_regularization = 0.0

        # Apply optional L2 regularization before normalization.
        if self.l2_regularization:
            total_regularization += self.l2_regularization * torch.sum(weights ** 2)

        if self.l2_normalize:
            weights = nn.functional.normalize(weights, p=2, dim=-1)

        similarities = torch.matmul(weights, weights.t())
        similarities = torch.diag_embed(similarities.new_zeros(similarities.size(0)))
        similarities_norm = torch.norm(similarities)

        total_regularization += self.spreadout_lambda * similarities_norm

        return total_regularization


class MatrixFactorizationModel(nn.Module):
    def __init__(self, num_users : int, 
                 num_items: int,
                 num_latent_factors: int,
                 personal_model : bool = True,
                 add_biases : bool = True,
                 l2_regularizer : float = 0.0,
                 spreadout_lambda : float = 0.0):
        super(MatrixFactorizationModel, self).__init__()
        
        self.num_users = num_users
        self.num_latent_factors = num_latent_factors
        self.num_items = num_items
        self.personal_model = personal_model
        self.add_biases = add_biases
        self.l2_regularizer = l2_regularizer
        self.spreadout_lambda = spreadout_lambda
        self.item_tensor = torch.empty((self.num_items, self.num_latent_factors))
        self.user_tensor = torch.empty((1, self.num_latent_factors))
        
        self.spread_out = EmbeddingSpreadoutRegularizer(spreadout_lambda=spreadout_lambda, l2_normalize=False, l2_regularization=l2_regularizer)
        
        #Initialize Layers
        self.global_layers = nn.ModuleList()
        self.local_layers = nn.ModuleList()

        self.item_embedding_layer = ItemEmbedding(self.num_items, self.num_latent_factors)
        self.global_layers.append(self.item_embedding_layer)
        self.spread_out = EmbeddingSpreadoutRegularizer(spreadout_lambda=spreadout_lambda, l2_normalize=False, l2_regularization=l2_regularizer)
        
        if self.personal_model:
            self.user_embedding_layer = UserEmbedding(num_latent_factors)
            self.local_layers.append(self.user_embedding_layer)
        else:
            self.user_embedding_layer = FullUserEmbedding(num_embeddings=num_users,
                                                     embedding_dim=num_latent_factors)
            self.local_layers.append(self.user_embedding_layer)
        
        if add_biases:
            if personal_model:
                self.user_bias_layer = AddBias(dim=(1,1))
                self.local_layers.append(self.user_bias_layer)
            else:
                self.user_bias_layer = AddBias(dim=(num_users, 1))
                self.local_layers.append(self.user_bias_layer)
            
            self.item_bias = AddBias(dim=(self.num_items, 1))
            self.global_layers.append(self.item_bias)
            self.global_bias = AddBias(dim=(1,1))
            self.global_layers.append(self.global_bias)
            
    def global_parameters(self):
        return [params for params in self.global_layers.parameters()]
    
    def local_parameters(self):
        return [params for params in self.local_layers.parameters()]
        
    
    def forward(self, user_input, item_input):
        user_embedding = self.user_embedding_layer(user_input)
        item_embedding = self.item_embedding_layer(item_input)
    
        
        flat_item_vec = item_embedding.view(-1, self.num_latent_factors)
        flat_user_vec = user_embedding.view(-1, self.num_latent_factors)
        
        prediction = torch.matmul(flat_user_vec, flat_item_vec.T).view(-1)
        if self.add_biases:
            flat_user_bias = self.user_bias_layer(user_input).view(-1)
            flat_item_bias = self.item_bias(item_input).view(-1)
            prediction += flat_user_bias + flat_item_bias
            prediction = prediction + self.global_bias(0)
        
        if self.l2_regularizer > 0.0:
            for param in self.parameters():
                l2_reg = self.l2_regularizer * torch.norm(param, p=2)**2
            prediction += l2_reg + self.spread_out(self.item_embedding_layer.item_embedding)
            
        return prediction
    
def build_reconstruction_model(num_users, num_items, num_latent_factors, personal_model=True, add_biases=True, l2_regularizer=0.0, spreadout_lambda=0.0):
    model = MatrixFactorizationModel(num_users, num_items, num_latent_factors, personal_model, add_biases, l2_regularizer, spreadout_lambda)
    global_params = model.global_parameters()
    local_params = model.local_parameters()
    return model, global_params, local_params


class ReconstructionAccuracyMetric:
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.sum = 0
        self.count = 0

    def update_state(self, y_true, y_pred):
        y_true = y_true.float()
        y_pred = y_pred.float()
        absolute_diffs = torch.abs(y_true - y_pred)
        example_accuracies = (absolute_diffs <= self.threshold).float()
        self.sum += example_accuracies.sum().item()
        self.count += example_accuracies.numel()

    def result(self):
        return self.sum / self.count if self.count > 0 else 0.0

    def reset_states(self):
        self.sum = 0
        self.count = 0
    

model, global_params, local_params = build_reconstruction_model(num_users=100, num_items=10, num_latent_factors=5, personal_model=True, add_biases=True, l2_regularizer=0.8, spreadout_lambda=0.0)
user_id = torch.tensor([1])
item_id = torch.tensor([9])
model_2, global_params2, local_params2 = build_reconstruction_model(num_users=100, num_items=10, num_latent_factors=5, personal_model=True, add_biases=True, l2_regularizer=0.8, spreadout_lambda=0.0)
# ex_loss = model(user_id, item_id)
# local_params = model.local_parameters()
# global_params = model.global_parameters()
# loss_val = nn.functional.mse_loss(ex_loss, torch.tensor([1.0]))
# #loss_val.backward()
# torch.autograd.grad(ex_loss, local_params[1], retain_graph=True)

model1_state_dict = model.state_dict()
model2_global_layers = [{k:v} for k,v in model_2.state_dict().items() if "global_layers" in k]

model.load_state_dict(model2_global_layers[0], strict=False)

for name, param in model.named_parameters():
    print("Name {} Param {}".format(name, param))
    # if param.requires_grad:
    #     print("Name {} {} Grad {}".format(name, param, param.grad))
        
