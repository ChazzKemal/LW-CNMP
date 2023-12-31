import torch
import torch.nn as nn
import torch.distributions as D
import numpy as np
from scipy.linalg import block_diag


class CNP(nn.Module):
    def __init__(self, in_shape, hidden_size, num_hidden_layers, min_std=0.1):
        super(CNP, self).__init__()
        self.d_x = in_shape[0]
        self.d_y = in_shape[1]

        self.encoder = []
        self.encoder.append(nn.Linear(self.d_x + self.d_y, hidden_size))
        self.encoder.append(nn.LayerNorm(hidden_size)) 
        self.encoder.append(nn.ReLU())
        for _ in range(num_hidden_layers - 1):
            self.encoder.append(nn.Linear(hidden_size, hidden_size))
            self.encoder.append(nn.LayerNorm(hidden_size)) 
            self.encoder.append(nn.ReLU())
        self.encoder.append(nn.Linear(hidden_size, hidden_size))
        self.encoder.append(nn.LayerNorm(hidden_size))
        self.encoder = nn.Sequential(*self.encoder)

        self.query = []
        self.query.append(nn.Linear(hidden_size + self.d_x, hidden_size))
        self.query.append(nn.ReLU())
        for _ in range(num_hidden_layers - 1):
            self.query.append(nn.Linear(hidden_size, hidden_size))
            self.query.append(nn.ReLU())
        self.query.append(nn.Linear(hidden_size, 2 * (self.d_y)))
        self.query = nn.Sequential(*self.query)

        self.min_std = min_std

    def nll_loss(self, observation, target, target_truth, observation_mask=None, target_mask=None):
        '''
        The original negative log-likelihood loss for training CNP.

        Parameters
        ----------
        observation : torch.Tensor
            (n_batch, n_context, d_x+d_y) sized tensor that contains context
            points.
            d_x: the number of query dimensions
            d_y: the number of target dimensions.
        target : torch.Tensor
            (n_batch, n_target, d_x) sized tensor that contains query dimensions
            of target (query) points.
            d_x: the number of query dimensions.
            note: n_context and n_target does not need to be the same size.
        target_truth : torch.Tensor
            (n_batch, n_target, d_y) sized tensor that contains target
            dimensions (i.e., prediction dimensions) of target points.
            d_y: the number of target dimensions
        observation_mask : torch.Tensor
            (n_batch, n_context) sized tensor indicating which entries should be
            used in aggregation. Used for batch input.
        target_mask : torch.Tensor
            (n_batch, n_target) sized tensor indicating which entries should be
            used for loss calculation. Used for batch input.

        Returns
        -------
        loss : torch.Tensor (float)
            The NLL loss.
        '''
        mean, std = self.forward(observation, target, observation_mask)
        #print(mean.shape,std.shape,target)
        dist = D.Normal(mean, std)
        nll = -dist.log_prob(target_truth)
        if target_mask is not None:
            # sum over the sequence (i.e. targets in the sequence)
            nll_masked = (nll * target_mask.unsqueeze(2)).sum(dim=1)
            # compute the number of entries for each batch entry
            nll_norm = target_mask.sum(dim=1).unsqueeze(1)
            # first normalize, then take an average over the batch and dimensions
            loss = (nll_masked / nll_norm).mean()
        else:
            loss = nll.mean()
        return loss

    def forward(self, observation, target, observation_mask=None):
        '''
        Forward pass of CNP.

        Parameters
        ----------
        observation : torch.Tensor
            (n_batch, n_context, d_x+d_y) sized tensor where d_x is the number
            of the query dimensions, d_y is the number of target dimensions.
        target : torch.Tensor
            (n_batch, n_target, d_x) sized tensor where d_x is the number of
            query dimensions. n_context and n_target does not need to be the
            same size.
        observation_mask : torch.Tensor
            (n_batch, n_context) sized tensor indicating which entries should be
            used in aggregation.

        Returns
        -------
        mean : torch.Tensor
            (n_batch, n_target, d_y) sized tensor containing the mean
            prediction.
        std : torch.Tensor
            (n_batch, n_target, d_y) sized tensor containing the standard
            deviation prediction.
        '''
        h = self.encode(observation)
        r = self.aggregate(h, observation_mask=observation_mask)
        h_cat = self.concatenate(r, target)
        query_out = self.decode(h_cat)
        mean = query_out[..., :(self.d_y)]
        logstd = query_out[..., (self.d_y):]
        std = torch.nn.functional.softplus(logstd) + self.min_std
        return mean, std

    def encode(self, observation):
        h = self.encoder(observation)
        return h

    def decode(self, h):
        o = self.query(h)
        return o

    def aggregate(self, h, observation_mask):
        # this operation is equivalent to taking mean but for
        # batched input with arbitrary lengths at each entry
        # the output should have (batch_size, dim) shape

        if observation_mask is not None:
            h = (h * observation_mask.unsqueeze(2)).sum(dim=1)  # mask unrelated entries and sum
            normalizer = observation_mask.sum(dim=1).unsqueeze(1)  # compute the number of entries for each batch entry
            r = h / normalizer  # normalize
        else:
            # if observation mask is none, we assume that all entries
            # in the batch has the same length
            r = h.mean(dim=1)
        return r

    def concatenate(self, r, target):
        num_target_points = target.shape[1]
        r = r.unsqueeze(1).repeat(1, num_target_points, 1)  # repeating the same r_avg for each target
        h_cat = torch.cat([r, target], dim=-1)
        return h_cat


class LocallyWeightedCNP(CNP):
    def __init__(self, in_shape, hidden_size, num_hidden_layers, min_std, weight_std=0.1):
        super(LocallyWeightedCNP, self).__init__(in_shape, hidden_size, num_hidden_layers, min_std)
        self.register_buffer("weight_std", torch.tensor(weight_std))

    def forward(self, observation, target, observation_mask=None, locally_weighted=True,aggregation_std=0.5):
        '''
        Parameters
        ----------
        observation : torch.Tensor
            (n_batch, n_context, d_x+d_y) sized tensor where d_x is the number
            of the query dimensions, d_y is the number of target dimensions.
        target : torch.Tensor
            (n_batch, n_target, d_x) sized tensor. n_context and n_target does
            not need to be the same size.
        observation_mask : torch.Tensor
            (n_batch, n_context) sized tensor indicating which entries should
            be used in aggregation.

        Returns
        -------
        mean : torch.Tensor
            (n_batch, n_target, d_y) sized tensor containing the mean
            prediction.
        std : torch.Tensor
            (n_batch, n_target, d_y) sized tensor containing the standard
            deviation prediction.
        '''

        h = self.encode(observation)
        if locally_weighted:
            r = self.aggregate(observation[..., 0], target[..., 0], h, observation_mask=observation_mask,aggregation_std=aggregation_std)
            h_cat = self.concatenate(r, target)
        else:
            r = super().aggregate(h, observation_mask)
            h_cat = super().concatenate(r, target)

        query_out = self.decode(h_cat)
        mean = query_out[..., :(self.d_y)]
        logstd = query_out[..., (self.d_y):]
        std = torch.nn.functional.softplus(logstd) + self.min_std
        return mean, std

    def aggregate(self, observation_t, target_t, h, observation_mask,aggregation_std=0.5):
        """
        Aggregate context point representation w.r.t. their proximity to target points.

        Parameters
        ----------
        observation_t : torch.Tensor
            (n_batch, n_context) shaped tensor containing the temporal
            dimension of context points
        target_t : torch.Tensor
            (n_batch, n_target) shaped tensor containing the temporal
            dimension of target points
        h : torch.Tensor
            (n_batch, n_context, n_dim) shaped tensor containing the encoded
            representation of context points.
        observation_mask : torch.Tensor
            (n_batch, n_context) sized tensor indicating which entries should
            be used in aggregation.

        Returns
        -------
        r : torch.Tensor
            (n_batch, n_target, n_dim) shaped tensor which contains the locally
            aggregated representation.
        """

        dist = D.Normal(target_t.unsqueeze(2), aggregation_std)
        weights = dist.log_prob(observation_t.unsqueeze(1).repeat(1, target_t.shape[1], 1)).exp()
        if observation_mask is not None:
            weights = weights * observation_mask.unsqueeze(1)
        weights = weights.to('cuda:0')
        weights_sum = weights.sum(dim=-1, keepdim=True).to('cuda:0')
        epsilon = 1e-8  # to prevent division by zero
        normalized_weights = weights / (weights_sum + epsilon)        
        r = normalized_weights @ h
        return r

    def concatenate(self, r, target):
        h_cat = torch.cat([r, target], dim=-1)
        return h_cat
class LW_VQCNP_2(CNP):
    def __init__(self, in_shape, hidden_size, num_hidden_layers, min_std, weight_std=0.1,memory_size=2,memory_partition=3/4):
        super(LW_VQCNP_2, self).__init__(in_shape, hidden_size, num_hidden_layers, min_std)
        self.register_buffer("weight_std", torch.tensor(weight_std))
        memory_dim = int(memory_partition * hidden_size)
        self.memory_dim = memory_dim
        self.memory = nn.Parameter(torch.randn(memory_size, memory_dim))
        self.loss=0
    
    def nll_loss(self, observation, target, target_truth, observation_mask=None, target_mask=None,primitive_index=None):
        '''
        The original negative log-likelihood loss for training CNP.

        Parameters
        ----------
        observation : torch.Tensor
            (n_batch, n_context, d_x+d_y) sized tensor that contains context
            points.
            d_x: the number of query dimensions
            d_y: the number of target dimensions.
        target : torch.Tensor
            (n_batch, n_target, d_x) sized tensor that contains query dimensions
            of target (query) points.
            d_x: the number of query dimensions.
            note: n_context and n_target does not need to be the same size.
        target_truth : torch.Tensor
            (n_batch, n_target, d_y) sized tensor that contains target
            dimensions (i.e., prediction dimensions) of target points.
            d_y: the number of target dimensions
        observation_mask : torch.Tensor
            (n_batch, n_context) sized tensor indicating which entries should be
            used in aggregation. Used for batch input.
        target_mask : torch.Tensor
            (n_batch, n_target) sized tensor indicating which entries should be
            used for loss calculation. Used for batch input.

        Returns
        -------
        loss : torch.Tensor (float)
            The NLL loss.
        '''
        mean, std = self.forward(observation, target, observation_mask,primitive_index=primitive_index)



        dist = D.Normal(mean, std)
        nll = -dist.log_prob(target_truth)
        if target_mask is not None:
            # sum over the sequence (i.e. targets in the sequence)
            nll_masked = (nll * target_mask.unsqueeze(2)).sum(dim=1)
            # compute the number of entries for each batch entry
            nll_norm = target_mask.sum(dim=1).unsqueeze(1)
            # first normalize, then take an average over the batch and dimensions
            loss = (nll_masked / nll_norm).mean()
            
        else:
            loss = nll.mean()
        loss += self.loss
        return loss



    def forward(self, observation, target, observation_mask=None, locally_weighted=True,aggregation_std=0.5,primitive_index=None):
        '''
        Parameters
        ----------
        observation : torch.Tensor
            (n_batch, n_context, d_x+d_y) sized tensor where d_x is the number
            of the query dimensions, d_y is the number of target dimensions.
        target : torch.Tensor
            (n_batch, n_target, d_x) sized tensor. n_context and n_target does
            not need to be the same size.
        observation_mask : torch.Tensor
            (n_batch, n_context) sized tensor indicating which entries should
            be used in aggregation.

        Returns
        -------
        mean : torch.Tensor
            (n_batch, n_target, d_y) sized tensor containing the mean
            prediction.
        std : torch.Tensor
            (n_batch, n_target, d_y) sized tensor containing the standard
            deviation prediction.
        '''
        self.loss=0
        h = self.encode(observation)
        if locally_weighted:
            r = self.aggregate(observation[..., 0], target[..., 0], h, observation_mask=observation_mask,aggregation_std=aggregation_std)

            r_memory = r[..., :self.memory_dim]
            r_residual = r[..., self.memory_dim:]
            selected_memory = self.memory_interaction(r_memory,primitive_index=primitive_index)
            if primitive_index != None:
                selected_memory = selected_memory.unsqueeze(1).repeat(1, r_residual.shape[1], 1)
                #calculate cosine similarity
                self.loss = torch.nn.functional.cosine_similarity(r_memory,selected_memory,dim=2).mean()
            #print(r_residual.shape, target.shape, selected_memory.shape)
            h_cat = self.concatenate(r_residual, target, selected_memory)
        else:
            r = super().aggregate(h, observation_mask)
            h_cat = super().concatenate(r, target)

        query_out = self.decode(h_cat)
        mean = query_out[..., :self.d_y]
        logstd = query_out[..., self.d_y:]
        std = torch.nn.functional.softplus(logstd) + self.min_std
        return mean, std
    
    def memory_interaction(self, r,primitive_index=None):
        """Interact with the memory using representation r."""
          # This should make r of shape [batch_size, 1, memory_dim]
        #print("Shape of r_memory after unsqueezing:", r.shape)
        memory_t = self.memory.transpose(0,1)
        memory_t = self.memory.t().unsqueeze(0).expand(r.size(0), -1, -1)
        #print("Shape of memory.t():", memory_t.shape)
        #print(primitive_index)
        memory_similarities = torch.bmm(r, memory_t)
        memory_similarities = memory_similarities.squeeze(1)
        if primitive_index == None:
            _ , indices = memory_similarities.max(dim=2)
            #print(indices)
        else:
            indices = primitive_index
        #print(indices)
        selected_memory = self.memory[indices]
        return selected_memory

    def aggregate(self, observation_t, target_t, h, observation_mask,aggregation_std=0.5):
        """
        Aggregate context point representation w.r.t. their proximity to target points.

        Parameters
        ----------
        observation_t : torch.Tensor
            (n_batch, n_context) shaped tensor containing the temporal
            dimension of context points
        target_t : torch.Tensor
            (n_batch, n_target) shaped tensor containing the temporal
            dimension of target points
        h : torch.Tensor
            (n_batch, n_context, n_dim) shaped tensor containing the encoded
            representation of context points.
        observation_mask : torch.Tensor
            (n_batch, n_context) sized tensor indicating which entries should
            be used in aggregation.

        Returns
        -------
        r : torch.Tensor
            (n_batch, n_target, n_dim) shaped tensor which contains the locally
            aggregated representation.
        """

        dist = D.Normal(target_t.unsqueeze(2), aggregation_std)
        weights = dist.log_prob(observation_t.unsqueeze(1).repeat(1, target_t.shape[1], 1)).exp()
        if observation_mask is not None:
            weights = weights * observation_mask.unsqueeze(1)
        weights = weights.to('cuda:0')
        weights_sum = weights.sum(dim=-1, keepdim=True).to('cuda:0')
        epsilon = 1e-8  # to prevent division by zero
        normalized_weights = weights / (weights_sum + epsilon)        
        r = normalized_weights @ h
        return r

    def concatenate(self, *tensors):
        
        h_cat = torch.cat(tensors, dim=-1)
        return h_cat

class LW_VQCNP(CNP):
    def __init__(self, in_shape, hidden_size, num_hidden_layers, min_std, weight_std=0.1,memory_size=2,memory_partition=3/4):
        super(LW_VQCNP, self).__init__(in_shape, hidden_size, num_hidden_layers, min_std)
        self.register_buffer("weight_std", torch.tensor(weight_std))
        memory_dim = int(memory_partition * hidden_size)
        self.memory_dim = memory_dim
        self.memory = nn.Parameter(torch.randn(memory_size, memory_dim))
    def forward(self, observation, target, observation_mask=None, locally_weighted=True,aggregation_std=0.5):
        '''
        Parameters
        ----------
        observation : torch.Tensor
            (n_batch, n_context, d_x+d_y) sized tensor where d_x is the number
            of the query dimensions, d_y is the number of target dimensions.
        target : torch.Tensor
            (n_batch, n_target, d_x) sized tensor. n_context and n_target does
            not need to be the same size.
        observation_mask : torch.Tensor
            (n_batch, n_context) sized tensor indicating which entries should
            be used in aggregation.

        Returns
        -------
        mean : torch.Tensor
            (n_batch, n_target, d_y) sized tensor containing the mean
            prediction.
        std : torch.Tensor
            (n_batch, n_target, d_y) sized tensor containing the standard
            deviation prediction.
        '''

        h = self.encode(observation)
        if locally_weighted:
            r = self.aggregate(observation[..., 0], target[..., 0], h, observation_mask=observation_mask,aggregation_std=aggregation_std)
            r_memory = r[..., :self.memory_dim]
            r_residual = r[..., self.memory_dim:]
            selected_memory = self.memory_interaction(r_memory)
            h_cat = self.concatenate(r_residual, target, selected_memory)
        else:
            r = super().aggregate(h, observation_mask)
            h_cat = super().concatenate(r, target)

        query_out = self.decode(h_cat)
        mean = query_out[..., :self.d_y]
        logstd = query_out[..., self.d_y:]
        std = torch.nn.functional.softplus(logstd) + self.min_std
        return mean, std
    
    def memory_interaction(self, r):
        """Interact with the memory using representation r."""
          # This should make r of shape [batch_size, 1, memory_dim]
        #print("Shape of r_memory after unsqueezing:", r.shape)
        memory_t = self.memory.transpose(0,1)
        memory_t = self.memory.t().unsqueeze(0).expand(r.size(0), -1, -1)
        #print("Shape of memory.t():", memory_t.shape)
        
        memory_similarities = torch.bmm(r, memory_t)
        memory_similarities = memory_similarities.squeeze(1)

        _ , indices = memory_similarities.max(dim=2)
        selected_memory = self.memory[indices]
        return selected_memory

    def aggregate(self, observation_t, target_t, h, observation_mask,aggregation_std=0.5):
        """
        Aggregate context point representation w.r.t. their proximity to target points.

        Parameters
        ----------
        observation_t : torch.Tensor
            (n_batch, n_context) shaped tensor containing the temporal
            dimension of context points
        target_t : torch.Tensor
            (n_batch, n_target) shaped tensor containing the temporal
            dimension of target points
        h : torch.Tensor
            (n_batch, n_context, n_dim) shaped tensor containing the encoded
            representation of context points.
        observation_mask : torch.Tensor
            (n_batch, n_context) sized tensor indicating which entries should
            be used in aggregation.

        Returns
        -------
        r : torch.Tensor
            (n_batch, n_target, n_dim) shaped tensor which contains the locally
            aggregated representation.
        """

        dist = D.Normal(target_t.unsqueeze(2), aggregation_std)
        weights = dist.log_prob(observation_t.unsqueeze(1).repeat(1, target_t.shape[1], 1)).exp()
        if observation_mask is not None:
            weights = weights * observation_mask.unsqueeze(1)
        weights = weights.to('cuda:0')
        weights_sum = weights.sum(dim=-1, keepdim=True).to('cuda:0')
        epsilon = 1e-8  # to prevent division by zero
        normalized_weights = weights / (weights_sum + epsilon)        
        r = normalized_weights @ h
        return r

    def concatenate(self, *tensors):
        h_cat = torch.cat(tensors, dim=-1)
        return h_cat


class ProMP:
    def __init__(self, n_dims, n_basis=10, kernel_range=(0, 1), kernel_width=None, amp=1.0, y_std=1e-4,
                 prior_width=1.0):
        self.n_basis = n_basis
        self.n_dims = n_dims
        self.mean = np.zeros(n_dims*n_basis)  # first n_basis are for first dim, next n_basis for second dim, etc.
        self.cov = np.eye(n_dims*n_basis) * prior_width
        self.kernel_means = np.linspace(kernel_range[0], kernel_range[1], n_basis).reshape(-1, 1)
        if kernel_width is None:
            kernel_width = (kernel_range[1] - kernel_range[0]) / n_basis
        self.kernel_width = kernel_width
        self.amp = amp
        self.y_std = y_std

    def learn_from_demonstrations(self, x, y):
        weights = []
        for x_i, y_i in zip(x, y):
            A = rbf(x_i, self.kernel_means, self.kernel_width, self.amp, self.n_dims)
            w_i = np.linalg.lstsq(A, y_i.T.reshape(-1), rcond=None)[0]
            weights.append(w_i)
        weights = np.stack(weights)
        self.mean = np.mean(weights, axis=0)
        self.cov = np.cov(weights, rowvar=False)

    def set_mean_cov(self, mean, cov):
        self.mean = mean
        self.cov = cov

    def condition(self, x, y, obs_noise=None):
        y = y.T.reshape(-1)
        A = rbf(x, self.kernel_means, self.kernel_width, self.amp, self.n_dims)
        if obs_noise is None:
            obs_noise = np.ones(y.shape[0])*self.y_std
        Sy = np.diag(obs_noise)
        temp = np.linalg.lstsq((A @ self.cov @ A.T + Sy).T, A @ self.cov.T, rcond=None)[0].T
        mean = self.mean + temp @ (y - A @ self.mean)
        cov = self.cov - temp @ A @ self.cov
        self.set_mean_cov(mean, cov)

    def query(self, t):
        A = rbf(t, self.kernel_means, self.kernel_width, self.amp, self.n_dims)
        y = A @ self.mean
        y_std = (A @ self.cov @ A.T + np.eye(y.shape[0])*self.y_std).diagonal()**0.5
        y = y.reshape(self.n_dims, -1).T
        y_std = y_std.reshape(self.n_dims, -1).T
        return y, y_std


def rbf(x, m, s, amp=1.0, dims=1):
    A = np.exp(-((x-m)**2)/(2*(s**2))).T
    A = amp * block_diag(*((A,) * dims))
    return A
