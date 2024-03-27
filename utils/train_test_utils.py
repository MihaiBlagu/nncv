import torch
import torch.nn.functional as F


def train(model, device, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()  

    # Avg loss per batch
    return total_loss / len(train_loader)  


def dice_score(outputs, masks, epsilon=1e-7):
    """
    Calculate the Dice score between predicted outputs and target masks.
    
    Args:
        outputs (torch.Tensor): Predicted outputs from the model. Shape: (batch_size, num_classes, height, width)
        masks (torch.Tensor): Target masks. Shape: (batch_size, 1, height, width)
        epsilon (float): Small constant to avoid division by zero.
        
    Returns:
        dice_score (torch.Tensor): Dice score.
    """
    intersection = torch.sum(outputs * masks, dim=(2, 3))
    union = torch.sum(outputs, dim=(2, 3)) + torch.sum(masks, dim=(2, 3))
    dice_scores = (2.0 * intersection) / (union + 1e-6) 

    return dice_scores.mean(dim=1)