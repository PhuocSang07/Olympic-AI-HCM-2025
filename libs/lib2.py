import torch
import pandas as pd
from tqdm import tqdm
from libs.lib1 import SubsetWithTransform, CustomTestDataset
from sklearn.metrics import accuracy_score
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split


def get_dataloaders(train_path, test_path, label_map, batch_size=32):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    transform = transforms.Compose([
            transforms.Resize((300, 300)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(degrees=15),
        
            transforms.ColorJitter(brightness=0.6, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAdjustSharpness(sharpness_factor=4, p=0.5),
            transforms.RandomAutocontrast(p=0.4),
            transforms.RandomEqualize(p=0.2),
        
            transforms.ToTensor(),
            transforms.GaussianBlur(kernel_size=(5,9), sigma=(0.1,5)),
            transforms.RandomErasing(
                p=0.3,
                scale=(0.02, 0.10),       # vùng che chiếm 2–10% diện tích
                ratio=(0.3, 3.3),         # giữ hình dạng hợp lý
                value='random'            # hoặc một màu cố định như 0
            ),
            transforms.Normalize(mean=mean, std=std),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    base_dataset = datasets.ImageFolder(root=train_path,transform=None)
    dataset_size = len(base_dataset)
    val_size     = int(0.2 * dataset_size)
    train_size   = dataset_size - val_size
    train_subset, val_subset = random_split(base_dataset, [train_size, val_size])

    train_dataset = SubsetWithTransform(train_subset, transform)
    val_dataset   = SubsetWithTransform(val_subset,   test_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

    submit_dataset = CustomTestDataset(test_path, transform=test_transform)
    test_loader = DataLoader(submit_dataset, batch_size=32, shuffle=False)

    class_names = base_dataset.classes
    print("Classes:", class_names)
    print("Original class_to_idx mapping:")
    print(base_dataset.class_to_idx)
    
    idx_to_class = {
        base_dataset.class_to_idx[orig]: label_map[orig]
        for orig in base_dataset.classes
    }
    return train_loader, val_loader, test_loader, base_dataset.classes, idx_to_class

def train(
        device, 
        model, 
        train_loader,
        val_loader, 
        epochs=50, 
        optimizer=None, 
        criterion=None, 
        scheduler=None, 
        idx_to_class=None
    ):
    model.to(device)
    best_val_loss = float('inf')  
    patience = 10  
    patience_counter = 0

    train_losses = []
    val_losses = []
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()
        for inputs, labels in tqdm(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            optimizer.step()
            optimizer.zero_grad()

            running_loss += loss.item()
        
        # Record training loss for this epoch
        epoch_train_loss = running_loss / len(train_loader)
        train_losses.append(epoch_train_loss) 
        print(f"Epoch {epoch+1}, Loss: {epoch_train_loss:.4f}", end=', ')

        # Validation phase
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                # Chuyển label và pred sang dạng label chuẩn hóa (string)
                pred_labels = [idx_to_class[p.item()] for p in preds]
                true_labels = [idx_to_class[l.item()] for l in labels]
            
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                all_preds.extend(pred_labels)
                all_labels.extend(true_labels)
        
        # Record validation loss for this epoch
        epoch_val_loss = val_loss / len(val_loader)
        val_losses.append(epoch_val_loss)
        
        acc = accuracy_score(all_labels, all_preds)
        print(f'Loss Valid: {epoch_val_loss:.4f}, Accuracy Valid: {acc:.4f}')
        
        # Early stopping check
        if epoch_val_loss < best_val_loss: #change here from > to <
            best_val_loss = epoch_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered!")
                break  
    scheduler.step()

def evaluate_model(model, dataloader, idx_to_class, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            # Chuyển label và pred sang dạng label chuẩn hóa (string)
            pred_labels = [idx_to_class[p.item()] for p in preds]
            true_labels = [idx_to_class[l.item()] for l in labels]

            all_preds.extend(pred_labels)
            all_labels.extend(true_labels)

    acc = accuracy_score(all_labels, all_preds)
    print(f'Accuracy: {acc:.4f}')
    return acc

def create_submission(model, dataloader, idx_to_class, device, path="output/results.csv"):
    model.eval()
    results = []
    file_names = []

    with torch.no_grad():
        for inputs, filenames in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            results.extend([idx_to_class[p.item()] for p in preds])
            file_names.extend(filenames)

    df = pd.DataFrame({
        "image_name": file_names,
        "label": results
    })
    df.to_csv(path, index=False)
    print(f"Results saved to {path}")
    return df