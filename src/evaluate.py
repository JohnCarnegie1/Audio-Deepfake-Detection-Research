import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

def evaluate_model(model, test_loader, device="cuda"):
    """
    Evaluate a trained SpikingJelly model on the test set.
    Resets all LIF neuron states before each batch to avoid size mismatches.

    Args:
        model (torch.nn.Module): Trained model
        test_loader (DataLoader): Test dataset loader
        device (str): 'cuda' or 'cpu'
    """
    model.eval()
    model.to(device)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)

            # --- Reset all LIF neuron states before forward pass ---
            for module in model.modules():
                if hasattr(module, "reset"):
                    module.reset()

            # Forward pass
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)

            # Collect results
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Safety check
    if len(all_labels) == 0 or len(all_preds) == 0:
        print("No predictions collected. Check your test_loader and dataset sizes.")
        return

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix - Test Set")
    plt.show()

    # Classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds))
