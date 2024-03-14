import torch

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
    smooth = 1e-5  # Small constant to prevent division by zero
    output = torch.argmax(output, dim=1).float()  # Assuming output is logits, applying argmax to get predicted class
    target = target.float()
    intersection = torch.sum(output * target)
    union = torch.sum(output) + torch.sum(target)
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice