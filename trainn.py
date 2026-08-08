import os
import random
import argparse
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

from model import SymUNet, get_model
from dataset import ImageRestorationDataset
from losses import get_loss_function
from metrics import RestorationMetrics

# Default paths
DEFAULT_DATA_DIR = 'train/train'
DRIVE_CHECKPOINTS_DIR = '/content/drive/MyDrive/KLA_project/checkpoints'
LOG_FILE_PATH = 'training_log.md'


def set_seed(seed=42):
    """
    Set random seeds across all libraries for deterministic, reproducible training.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"🔒 Random seed set to: {seed} (Reproducible mode)")


def verify_drive_access(drive_dir):
    """
    Fail-fast check: Verify Google Drive is mounted and writable before starting training.
    """
    try:
        os.makedirs(drive_dir, exist_ok=True)
        test_file = os.path.join(drive_dir, '.drive_write_test.tmp')
        with open(test_file, 'w') as f:
            f.write('ok')
        os.remove(test_file)
        print(f"✓ Google Drive write access verified: {drive_dir}")
        return True
    except Exception as e:
        print(f"\n❌ [CRITICAL WARNING] Cannot write to Google Drive path: {drive_dir}")
        print(f"   Reason: {e}")
        print("   Make sure you ran `from google.colab import drive; drive.mount('/content/drive')` in Colab!\n")
        return False


def log_experiment_to_md(stage, run_number, epoch, loss_name, train_loss, val_loss, metrics_dict, lr, is_best):
    """
    Auto-appends epoch training and validation metrics (PSNR, SSIM, LPIPS) directly to training_log.md.
    """
    header_needed = not os.path.exists(LOG_FILE_PATH) or os.path.getsize(LOG_FILE_PATH) == 0
    with open(LOG_FILE_PATH, 'a', encoding='utf-8') as f:
        if header_needed:
            f.write("# 📋 KLA Image Restoration — Training Experiment Log\n\n")
            f.write("| Timestamp | Stage | Run | Epoch | Loss Name | Train Loss | Val Loss | Val PSNR (dB) | Val SSIM | Val LPIPS | LR | Best? |\n")
            f.write("| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        best_marker = "🌟 BEST" if is_best else ""
        lpips_str = f"{metrics_dict['lpips']:.4f}" if metrics_dict.get('lpips') is not None else "N/A"
        
        f.write(f"| {timestamp} | {stage} | {run_number} | {epoch} | {loss_name} | {train_loss:.4f} | {val_loss:.4f} | {metrics_dict['psnr']:.4f} | {metrics_dict['ssim']:.4f} | {lpips_str} | {lr:.6f} | {best_marker} |\n")


def save_smart_checkpoint(model, optimizer, scheduler, epoch, train_loss, val_loss, metrics_dict, is_best, stage="stage_2", run_number=1, loss_name="l1", use_drive=False, save_all=False):
    """
    Smart Checkpoint Manager:
      - Always saves '{stage}_{loss_name}_latest.pth'
      - Saves '{stage}_{loss_name}_best.pth' when highest Val PSNR is reached
      - Keeps Google Drive storage lean and prevents quota exhaustion.
    """
    os.makedirs('checkpoints', exist_ok=True)
    if use_drive:
        os.makedirs(DRIVE_CHECKPOINTS_DIR, exist_ok=True)

    prefix = f"{stage}_{loss_name}_run{run_number}"

    checkpoint_data = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'val_psnr': metrics_dict['psnr'],
        'val_ssim': metrics_dict['ssim'],
        'val_lpips': metrics_dict.get('lpips'),
        'loss_name': loss_name,
        'stage': stage,
        'run_number': run_number,
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
    }

    # 1. Save LATEST checkpoint
    latest_local = os.path.join('checkpoints', f'{prefix}_latest.pth')
    torch.save(checkpoint_data, latest_local)

    if use_drive:
        latest_drive = os.path.join(DRIVE_CHECKPOINTS_DIR, f'{prefix}_latest.pth')
        torch.save(checkpoint_data, latest_drive)

    # 2. Save BEST checkpoint
    if is_best:
        best_local = os.path.join('checkpoints', f'{prefix}_best.pth')
        torch.save(checkpoint_data, best_local)
        print(f"🌟 New BEST checkpoint saved! Val PSNR: {metrics_dict['psnr']:.4f} dB -> {best_local}")
        if use_drive:
            best_drive = os.path.join(DRIVE_CHECKPOINTS_DIR, f'{prefix}_best.pth')
            torch.save(checkpoint_data, best_drive)

    # 3. Optional: Save specific epoch archive
    if save_all:
        epoch_local = os.path.join('checkpoints', f"{prefix}_epoch{epoch}_psnr{metrics_dict['psnr']:.2f}.pth")
        torch.save(checkpoint_data, epoch_local)
        if use_drive:
            epoch_drive = os.path.join(DRIVE_CHECKPOINTS_DIR, f"{prefix}_epoch{epoch}_psnr{metrics_dict['psnr']:.2f}.pth")
            torch.save(checkpoint_data, epoch_drive)


def evaluate_validation(model, val_loader, criterion, metric_evaluator, device):
    """
    Runs validation with model.eval() and torch.no_grad(), computing:
      - Validation Loss
      - Validation PSNR (dB)
      - Validation SSIM
      - Validation LPIPS (if available)
    """
    model.eval()
    val_loss = 0.0
    running_psnr = 0.0
    running_ssim = 0.0
    running_lpips = 0.0
    n_batches = len(val_loader)

    with torch.no_grad():
        for noisy, gt in val_loader:
            noisy, gt = noisy.to(device), gt.to(device)
            output = model(noisy)

            # Resize output 128x128 -> 256x256 to match GT if needed
            if output.shape != gt.shape:
                output = torch.nn.functional.interpolate(output, size=gt.shape[2:], mode='bilinear', align_corners=False)

            loss = criterion(output, gt)
            val_loss += loss.item()

            batch_m = metric_evaluator.compute_batch(output, gt)
            running_psnr += batch_m['psnr']
            running_ssim += batch_m['ssim']
            if batch_m.get('lpips') is not None:
                running_lpips += batch_m['lpips']

    avg_val_loss = val_loss / max(n_batches, 1)
    avg_metrics = {
        'psnr': running_psnr / max(n_batches, 1),
        'ssim': running_ssim / max(n_batches, 1),
        'lpips': (running_lpips / max(n_batches, 1)) if metric_evaluator.compute_lpips else None
    }

    return avg_val_loss, avg_metrics


def train(
    epochs=9,
    batch_size=8,
    lr=0.001,
    stage="stage_2",
    run_number=1,
    data_dir=DEFAULT_DATA_DIR,
    use_drive=False,
    resume_path=None,
    scheduler_type="plateau",
    loss_type="l1",
    seed=42,
    num_workers=2,
    preload_ram=True,
    save_all_epochs=False,
    alpha_zhao=0.85,
    w_gfl=0.10,
    base_channels=64,
    model_type="unet"
):
    # Set seed
    set_seed(seed)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Device: {device} | Stage: {stage} (Run #{run_number}) | Model: {model_type.upper()} | Loss: {loss_type.upper()}")

    # Fail-fast Drive check
    if use_drive:
        drive_ok = verify_drive_access(DRIVE_CHECKPOINTS_DIR)
        if not drive_ok:
            print("⚠ Proceeding with local checkpointing only until Drive is mounted.")

    # 1. Datasets & DataLoaders (Train 90% / Val 10%)
    train_dataset = ImageRestorationDataset(
        data_dir=data_dir,
        split_ratio=0.9,
        is_val=False,
        preload_to_ram=preload_ram
    )
    val_dataset = ImageRestorationDataset(
        data_dir=data_dir,
        split_ratio=0.9,
        is_val=True,
        preload_to_ram=preload_ram
    )

    loader_kwargs = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'pin_memory': (device.type == 'cuda'),
    }
    if num_workers > 0:
        loader_kwargs['persistent_workers'] = True
        loader_kwargs['prefetch_factor'] = 2

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    # 2. Model Setup via Factory
    model = get_model(model_type, in_channels=1, out_channels=1, base_channels=base_channels).to(device)

    # 3. Metric Evaluator
    metric_evaluator = RestorationMetrics(device=device, compute_lpips=True)

    # 4. Loss Function via Factory
    criterion = get_loss_function(loss_type, alpha_zhao=alpha_zhao, w_gfl=w_gfl).to(device)

    # 5. Optimizer & Scheduler
    optimizer = optim.Adam(model.parameters(), lr=lr)

    scheduler = None
    if scheduler_type == "plateau":
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
        print("📈 Scheduler: ReduceLROnPlateau (tracking Val PSNR, factor=0.5, patience=2)")
    elif scheduler_type == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        print(f"📈 Scheduler: CosineAnnealingLR (T_max={epochs}, eta_min=1e-6)")
    else:
        print("📈 Scheduler: Fixed Learning Rate")

    # 6. Checkpoint Resume Logic
    start_epoch = 0
    best_val_psnr = -float('inf')

    if resume_path and os.path.exists(resume_path):
        print(f"\n📂 Resuming from Checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)

        model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))

        if isinstance(checkpoint, dict) and 'optimizer_state_dict' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print("✓ Optimizer state restored successfully")
            except Exception as e:
                print(f"⚠ Could not restore optimizer: {e}")

        if scheduler and isinstance(checkpoint, dict) and checkpoint.get('scheduler_state_dict'):
            try:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                print("✓ Scheduler state restored")
            except Exception as e:
                print(f"⚠ Could not restore scheduler: {e}")

        start_epoch = checkpoint.get('epoch', 0) if isinstance(checkpoint, dict) else 0
        best_val_psnr = checkpoint.get('val_psnr', -float('inf')) if isinstance(checkpoint, dict) else -float('inf')
        print(f"✓ Resumed from Epoch {start_epoch} (Prior Best Val PSNR: {best_val_psnr:.2f} dB)\n")

    total_epochs = start_epoch + epochs
    print(f"⚡ Ready! Training on {len(train_dataset)} images, Validating on {len(val_dataset)} images (Epochs {start_epoch + 1} to {total_epochs})...\n")

    # 7. Training Loop
    for epoch in range(start_epoch, total_epochs):
        model.train()
        train_loss = 0.0

        for batch_idx, (noisy, gt) in enumerate(train_loader):
            noisy, gt = noisy.to(device), gt.to(device)

            optimizer.zero_grad()
            output = model(noisy)

            # The model naturally outputs the 2x upscaled image via PixelShuffle.
            # No manual bilinear interpolation needed here anymore.
            loss = criterion(output, gt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()

            if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(train_loader):
                curr_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch [{epoch+1}/{total_epochs}] | Batch [{batch_idx+1}/{len(train_loader)}] | Train Loss: {loss.item():.4f} | LR: {curr_lr:.6f}")

        avg_train_loss = train_loss / len(train_loader)

        # 8. Validation Pass with PSNR, SSIM, and LPIPS
        val_loss, val_metrics = evaluate_validation(model, val_loader, criterion, metric_evaluator, device)
        curr_lr = optimizer.param_groups[0]['lr']

        is_best = val_metrics['psnr'] > best_val_psnr
        if is_best:
            best_val_psnr = val_metrics['psnr']

        # Step Scheduler (tracking Val PSNR if plateau)
        if scheduler:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_metrics['psnr'])
            else:
                scheduler.step()

        lpips_display = f" | Val LPIPS: {val_metrics['lpips']:.4f}" if val_metrics.get('lpips') is not None else ""
        print("-" * 80)
        print(f"✅ Epoch {epoch+1}/{total_epochs} Complete | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val PSNR: {val_metrics['psnr']:.4f} dB | Val SSIM: {val_metrics['ssim']:.4f}{lpips_display}")
        print("-" * 80)

        # Auto-append to training_log.md
        log_experiment_to_md(
            stage=stage,
            run_number=run_number,
            epoch=epoch+1,
            loss_name=loss_type,
            train_loss=avg_train_loss,
            val_loss=val_loss,
            metrics_dict=val_metrics,
            lr=curr_lr,
            is_best=is_best
        )

        # Save Smart Checkpoint
        save_smart_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch+1,
            train_loss=avg_train_loss,
            val_loss=val_loss,
            metrics_dict=val_metrics,
            is_best=is_best,
            stage=stage,
            run_number=run_number,
            loss_name=loss_type,
            use_drive=use_drive,
            save_all=save_all_epochs
        )

    print(f"\n🎉 Training complete! Final Val PSNR: {val_metrics['psnr']:.4f} dB | Best Val PSNR: {best_val_psnr:.4f} dB\n")
    return val_metrics['psnr']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Modular Ablation Training Script for KLA Restoration")
    parser.add_argument("--epochs", type=int, default=7, help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--stage", type=str, default="stage_2")
    parser.add_argument("--run_number", type=int, default=1)
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--use_drive", action="store_true", help="Backup checkpoints to Google Drive")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint .pth to resume from")
    parser.add_argument("--scheduler", type=str, default="plateau", choices=["plateau", "cosine", "none"])
    parser.add_argument("--loss", type=str, default="charbonnier",
                        choices=["charbonnier", "compound", "l1", "mse", "msssim", "baseline"],
                        help="Loss function (charbonnier / compound / l1 / mse / msssim / baseline)")
    parser.add_argument("--alpha_zhao", type=float, default=0.90, help="MS-SSIM ratio in Zhao mix (default 0.90)")
    parser.add_argument("--w_gfl", type=float, default=0.10, help="Weight for GFL frequency loss (default 0.10)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--no_preload_ram", action="store_true", help="Disable RAM dataset preloading")
    parser.add_argument("--save_all_epochs", action="store_true", help="Save separate .pth for every single epoch")
    parser.add_argument("--base_channels", type=int, default=64, help="Base channel count for SymUNet (use 32 or 16 for efficiency sweep)")
    parser.add_argument("--model", type=str, default="symunet", choices=["symunet"], help="Model architecture (defaults to symunet)")
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        stage=args.stage,
        run_number=args.run_number,
        data_dir=args.data_dir,
        use_drive=args.use_drive,
        resume_path=args.resume,
        scheduler_type=args.scheduler,
        loss_type=args.loss,
        seed=args.seed,
        num_workers=args.num_workers,
        preload_ram=(not args.no_preload_ram),
        save_all_epochs=args.save_all_epochs,
        alpha_zhao=args.alpha_zhao,
        w_gfl=args.w_gfl,
        base_channels=args.base_channels,
        model_type=args.model
    )
