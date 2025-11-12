import torch
import torch.nn as nn

def train(model, train_loader, val_loader, epochs=70, lr=1e-4, device="cuda"):
    """
    Train a spiking CNN model with proper neuron resets for SpikingJelly.
    
    Args:
        model: PyTorch model (SpikingCNN)
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        epochs: number of training epochs
        lr: learning rate
        device: 'cuda' or 'cpu'
    
    Returns:
        trained model, train_losses, val_losses, val_accuracies
    """
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    train_losses, val_losses, val_accuracies = [], [], []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for data, label in train_loader:
            data, label = data.to(device), label.to(device)

            # Reset all LIF neuron states before forward pass
            for module in model.modules():
                if hasattr(module, "reset"):
                    module.reset()

            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, label)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for data, label in val_loader:
                data, label = data.to(device), label.to(device)

                # Reset neuron states during validation too
                for module in model.modules():
                    if hasattr(module, "reset"):
                        module.reset()

                output = model(data)
                val_loss += loss_fn(output, label).item()
                _, predicted = torch.max(output, 1)
                correct += (predicted == label).sum().item()
                total += label.size(0)

        avg_val_loss = val_loss / len(val_loader)
        accuracy = correct / total
        val_losses.append(avg_val_loss)
        val_accuracies.append(accuracy)

        print(f"Epoch [{epoch+1}/{epochs}], "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, "
              f"Val Accuracy: {accuracy:.4f}")

    return model, train_losses, val_losses, val_accuracies
