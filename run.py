from data import get_cifar10
import torch
import torch.nn as nn
import model
import json
import wandb

class ViT(nn.Module):
    def __init__(self, f, *args, **kwargs):
        super().__init__(*args, **kwargs)
        height = f["image"]["height"]
        width = f["image"]["width"]
        channel = f["image"]["channels"]
        patch = f["image"]["patch"]

        self.d_model = f["transformer"]["d_model"]
        heads = f["transformer"]["heads"]
        self.dropout = f["transformer"]["dropout"]
        d_ff = f["transformer"]["d_ff"]
        n_layers = f["transformer"]["N"]

        embed = model.PatchInputEmbedding(height=height, width=width, 
                                          d_model=self.d_model, channel=channel, patch=patch)
        layers = nn.ModuleList([
            model.Encoder(self.d_model, self.dropout,
                          model.MultiHeadAttention(self.d_model, heads, self.dropout),
                          model.FeedForwardNetwork(self.d_model, d_ff, self.dropout))
                          for _ in range(n_layers)
        ])

        self.transformer = model.ViT_Block(embed=embed, encoder=layers, d_model=self.d_model, 
                                           num_classes=f["classifier"]["num_classes"], dropout=self.dropout)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    def forward(self, x):
        return self.transformer(x)
    
with open('config.json') as f:
    config = json.load(f)

def train():
    wandb.init(
        project="vit-cifar10",
        config=config,
    )
 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
    train_loader, val_loader = get_cifar10(
        batch_size=config["training"]["batch_size"]
    )
 
    vit = ViT(config).to(device)
    wandb.watch(vit, log="gradients", log_freq=100)
 
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        vit.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
 
    epochs        = config["training"]["epochs"]
    warmup_epochs = config["training"]["warmup_epochs"]
 
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        return 1.0
 
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
 
    for epoch in range(epochs):
 
        vit.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
 
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
 
            optimizer.zero_grad()
            logits = vit(images)
            loss   = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vit.parameters(), max_norm=1.0)
            optimizer.step()
 
            train_loss    += loss.item() * images.size(0)
            train_correct += (logits.argmax(dim=-1) == labels).sum().item()
            train_total   += images.size(0)
 
        scheduler.step()
 
        train_loss /= train_total
        train_acc   = train_correct / train_total
 
        vit.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
 
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits = vit(images)
                loss   = criterion(logits, labels)
 
                val_loss    += loss.item() * images.size(0)
                val_correct += (logits.argmax(dim=-1) == labels).sum().item()
                val_total   += images.size(0)
 
        val_loss /= val_total
        val_acc   = val_correct / val_total
 
        wandb.log({
            "epoch":  epoch + 1,
            "train/loss": train_loss,
            "train/acc": train_acc,
            "val/loss": val_loss,
            "val/acc":  val_acc,
            "lr":  scheduler.get_last_lr()[0],
        })
 
        print(f"Epoch {epoch+1:03d} | "
              f"train loss {train_loss:.4f} acc {train_acc:.3f} | "
              f"val loss {val_loss:.4f} acc {val_acc:.3f}")
 
    wandb.finish()
 
if __name__ == "__main__":
    train()
 
