from src import SpectrogramDataset, create_dataloaders, SpikingCNN, train, get_data_transforms


if __name__ == "__main__":
    # --- Load data ---
    transformed = get_data_transforms()
    dataset = SpectrogramDataset(
        spectrogram_folder="spectrograms_new_may", 
        label_file_path="label.txt", 
        label_map={'bonafide': 0, 'spoof': 1},
        transform=transformed
    )

    train_loader, val_loader, test_loader = create_dataloaders(dataset, batch_size=4)

    # --- Build model ---
    model = SpikingCNN()

    # --- Train ---
    trained_model, train_losses, val_losses, val_accuracies = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=70,           
        lr=1e-4,
        device="cuda"        # ensure GPU usage
    )

    print("Training complete!")
