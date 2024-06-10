import torch
#for example
checkpoint_path = 'results/mnist_micro_rid/checkpoints/checkpoint_7.pth'

checkpoint = torch.load(checkpoint_path)


model_state_dict = checkpoint['model']
total_params = sum(param.numel() for param in model_state_dict.values())

print(f'Numb of params: {total_params}')
