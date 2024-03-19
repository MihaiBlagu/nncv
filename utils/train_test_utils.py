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


def dice_score(output, target):
    # Small constant to prevent division by zero
    eps = 1e-5  
    # Smoothing factor
    smooth = 1.
    
    # # Convert logits to probabilities
    # output_probs = F.softmax(output, dim=1)  # Assuming output is logits
    
    # Apply argmax to get predicted class
    output = torch.argmax(output, dim=1).float() 
    
    # Convert target to float
    target = target.float()
    
    # Calculate intersection and union
    intersection = torch.sum(output * target)
    sum_output = torch.sum(output)
    sum_target = torch.sum(target)
    union = sum_output + sum_target
    
    # Calculate Dice score
    dice = (2.0 * intersection) / (union + eps)
    return dice