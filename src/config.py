import torch

ENV_NAME = "CartPole-v1"
HIDDEN_SIZE = 16
N_EPISODES  = 1000
GAMMA       = 1.0
LR          = 1e-2
PRINT_EVERY = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
