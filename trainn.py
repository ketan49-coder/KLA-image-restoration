import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from model import UNet
from dataset import ImageRestorationDataset
from losses import CombinedLoss
import os

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Dataset and DataLoader
    dataset = ImageRestorationDataset('train/train')
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

    # Model, loss, optimizer
    model = UNet(in_channels=1, out_channels=1).to(device)
    criterion = CombinedLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(f"Training on {len(dataset)} images")
    print("Starting training...")

    # Train for 1 epoch
    model.train()
    total_loss = 0
    for batch_idx, (noisy, gt) in enumerate(train_loader):
        noisy, gt = noisy.to(device), gt.to(device)

        optimizer.zero_grad()
        output = model(noisy)

        # Resize output to match gt if needed
        if output.shape != gt.shape:
            output = torch.nn.functional.interpolate(output, size=gt.shape[2:], mode='bilinear', align_corners=False)

        loss = criterion(output, gt)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 100 == 0:
            print(f"Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(train_loader)
    print(f"\nTraining complete!")
    print(f"Average Loss: {avg_loss:.4f}")

    # Save model
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(model.state_dict(), 'checkpoints/unet_model.pth')
    print("Model saved to checkpoints/unet_model.pth")

if __name__ == '__main__':
    train()
