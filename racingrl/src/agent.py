import torch.nn as nn
import torch
import numpy as np
import typing as tt

from ptan.agent import BaseAgent
from ptan.actions import ActionSelector


States = tt.List[np.ndarray] | np.ndarray
AgentStates = tt.List[tt.Any]


class NormalSamplingActionSelector(ActionSelector):
    def __call__(
            self, 
            actions_mu: torch.Tensor, 
            actions_var: torch.Tensor
        ) -> torch.Tensor:

        return torch.normal(actions_mu, actions_var)
    

class Agent(BaseAgent):
    def __init__(self, net, optimizer, config):
        super().__init__()

        self.net = net
        self.optimizer = optimizer

        self.device =       config['device']
        self.entropy_beta = config['entropy_beta']
        self.gamma =        config['gamma']
        self.reward_steps = config['reward_steps']

        self.action_selector = NormalSamplingActionSelector()

        self.net.to(self.device)
        
        self.reward_tracker = dict()

    @torch.no_grad
    def __call__(
            self, 
            states: States, 
            agent_states: AgentStates = None
    ) -> tt.Tuple[np.ndarray, AgentStates]:
        states_t = torch.tensor(np.array(states)).to(self.device)
        actions_mu, actions_var = self.net.forward_policy(states_t)
        action_vals = self.action_selector(actions_mu, actions_var)
        return action_vals.cpu().numpy(), agent_states


    def update_policy(self, batch):
        # update will log results to reward_tracker
        self.reward_tracker.clear()

        actions, states, Qs = self.unpack_batch(batch)

        self.optimizer.zero_grad()
        loss = self.calc_loss(actions, states, Qs)
        loss.backward()
        self.optimizer.step()

        return self.reward_tracker

    def unpack_batch(self, batch):
        actions = []
        states = []
        rewards = []
        last_states = []
        non_last_states_idx = [] # states for which to calculate succeeding Vs
        for idx, exp in enumerate(batch):
            actions.append(exp.action)
            states.append(exp.state)

            # discounted reward for first reward_steps steps
            rewards.append(exp.reward)

            if exp.last_state is not None:
                non_last_states_idx.append(idx)
                last_states.append(exp.last_state)

        # convert to numpy afterwards to avoid inefficient apends
        actions_t = torch.tensor(np.array(actions), dtype=torch.float32).to(self.device)
        states_t = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        last_states_t = torch.tensor(np.array(last_states), dtype=torch.float32).to(self.device)
        rewards_t = torch.tensor(np.array(rewards), dtype=torch.float32).to(self.device)

        if non_last_states_idx:
            # calculate Bellman approximation of value for the end of non-terminal states
            # and add them to observed rewards
            pred_vals = self.net(last_states_t)[2].detach() # single forward pass
            pred_vals *= self.gamma ** self.reward_steps
            rewards_t[non_last_states_idx] += pred_vals[...,0] # ommit batch dimension

        return actions_t, states_t, rewards_t

    def calc_loss(self, actions, states, Qs):
        pred_actions_mu, pred_actions_var, pred_values = self.net(states)

        action_loss = -self.policy_loss(pred_actions_mu, pred_actions_var, pred_values[...,0], Qs, actions)
        value_loss = nn.functional.mse_loss(Qs, pred_values[...,0])
        entropy_bonus = -self.entropy(pred_actions_var)

        loss = action_loss + value_loss + entropy_bonus * self.entropy_beta
        # print(f'Policy loss: {-action_loss.item()}')
        # print(f'Value loss: {value_loss.item()}')
        # print(f'Entropy bonus: {-entropy_bonus.item()}')
        # print(f'Loss: {loss.item()}')

        # store logging info
        action_loss.backward(retain_graph=True)
        grads = np.concatenate([p.grad.data.cpu().numpy().flatten()
                        for p in self.net.parameters()
                        if p.grad is not None])
        self.reward_tracker['policy_loss_grads'] = grads
        self.reward_tracker['policy_loss'] = action_loss.detach().cpu().numpy()
        self.reward_tracker['value_loss'] = value_loss.detach().cpu().numpy()
        self.reward_tracker['entropy_bonus'] = entropy_bonus.detach().cpu().numpy()

        return loss


    def policy_loss(
            self, 
            pred_actions_mu: torch.Tensor, 
            pred_actions_var: torch.Tensor,
            Vs: torch.Tensor, 
            Qs: torch.Tensor, 
            actions: torch.Tensor
    ) -> torch.Tensor:
        """Policy loss for normaly distributed actions"""
        
        # log probability of observed actions in normal dist with predicted params
        p1 = -torch.log(torch.sqrt(2 * torch.pi * pred_actions_var))
        p2 = - torch.square(actions - pred_actions_mu) / (2 * pred_actions_var)
        actions_logprob = p1 + p2
        
        # subtract mean action value from Q-vals to reduce variance
        As = Qs - Vs

        # action log prob is the sum of log probs across action space
        obs_wise_loss = \
            (As.unsqueeze(dim=-1) * actions_logprob).sum(dim=1)
        loss = obs_wise_loss.mean()

        # log action values
        self.reward_tracker['As'] = As.detach().cpu().numpy().mean().item()
        self.reward_tracker['Qs'] = Qs.detach().cpu().numpy().mean().item()
        self.reward_tracker['Vs'] = Vs.detach().cpu().numpy().mean().item()

        return loss
    
    @staticmethod
    def entropy(pred_actions_var: torch.Tensor) -> torch.Tensor:
        """Entropy of observed normally distributed predictions"""
        ent = (torch.log(2*torch.pi*pred_actions_var) + 1) / 2

        return ent.mean()
        