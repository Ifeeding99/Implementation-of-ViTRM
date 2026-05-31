import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from vitrm import ViTRM
from torchmetrics.classification.accuracy import Accuracy
import tqdm
import einops
import numpy as np
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from dataprocessing import get_brain_tumor_data_loader


def train_ViTRM(model, train_loader, val_loader, n_epochs, 
                num_classes, n_supevision_steps, lr=1e-3, threshold_tau=0.5):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    criterion_classification = nn.CrossEntropyLoss()
    criterion_halting = nn.BCELoss()
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer, max_lr=lr,
                                                    epochs=n_epochs, steps_per_epoch=len(train_loader),
                                                    pct_start=0.05, anneal_strategy='cos')
    train_acc = Accuracy(task='multiclass', num_classes=num_classes)
    val_acc = Accuracy(task='multiclass', num_classes=num_classes)

    model = model.to(device)
    ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.999)) # used for stability in evaluation
    ema_model = ema_model.to(device)
    train_acc = train_acc.to(device)
    val_acc = val_acc.to(device)
    for epoch in range(n_epochs):
        train_bar = tqdm.tqdm(train_loader)
        val_bar = tqdm.tqdm(val_loader)
        train_losses = []
        val_losses = []
        print(f'Epoch {epoch+1}/{n_epochs}')
        model.train(True)
        for idx,(x,target) in enumerate(train_bar):
            x = x.to(device)
            target = target.to(device)
            y = einops.repeat(model.y0, '1 1 emb_dim -> B 1 emb_dim', B=x.shape[0])
            z = einops.repeat(model.z0, '1 k emb_dim -> B k emb_dim', B=x.shape[0], k=model.k)
            for step in range(n_supevision_steps):
                patches = model.patch_embeddings(x)
                classes, h_prob, y, z = model(patches, y, z)
                loss_classification_step = criterion_classification(classes, target)
                correctly_predicted = (torch.argmax(classes, dim=-1) == target).float() # assuming target has 0,1,2,... as labels
                halting_loss = criterion_halting(h_prob.squeeze(1), correctly_predicted)
                loss_step = loss_classification_step + halting_loss

                optimizer.zero_grad()
                loss_step.backward()
                optimizer.step()

                ema_model.update_parameters(model)

                y = y.detach()
                z = z.detach()

                if h_prob.mean() > threshold_tau:
                    break

            scheduler.step()

            current_loss = loss_step.item()
            train_losses.append(current_loss)
            train_acc.update(classes.detach(), target)
            train_bar.set_description('Training')
            train_bar.set_postfix(batch_loss = round(current_loss, 3))

        ema_model.eval()
        with torch.no_grad():
            for idx, (x, target) in enumerate(val_bar):
                x = x.to(device)
                target = target.to(device)
                y = einops.repeat(ema_model.module.y0, '1 1 emb_dim -> B 1 emb_dim', B=x.shape[0])
                z = einops.repeat(ema_model.module.z0, '1 k emb_dim -> B k emb_dim', B=x.shape[0], k=ema_model.module.k)
                patches = ema_model.module.patch_embeddings(x)
                classes, h_prob, y, z = ema_model(patches, y, z)
                loss_classification_step = criterion_classification(classes, target)
                correctly_predicted = (torch.argmax(classes, dim=-1) == target).float() # assuming target has 0,1,2,... as labels
                halting_loss = criterion_halting(h_prob.squeeze(1), correctly_predicted)
                loss_step = loss_classification_step + halting_loss

                current_loss = loss_step.item()
                val_losses.append(current_loss)
                val_acc.update(classes.detach(), target)
                val_bar.set_description('Validation')
                val_bar.set_postfix(batch_loss = round(current_loss, 3))

        
        print(f'Average train loss: {np.array(train_losses).mean():.3f}, average train accuracy: {train_acc.compute():.3f}')
        train_acc.reset()

        print(f'Average validation loss: {np.array(val_losses).mean():.3f}, average validation accuracy: {val_acc.compute():.3f}')
        val_acc.reset()


if __name__ == '__main__':
    train_loader = get_brain_tumor_data_loader(path_to_dataset='/home/ifeeding99/Downloads/brain-tumor-dataset/Training',
                                                batch_size=1, t_train=True, shuffle=True)
    
    val_loader = get_brain_tumor_data_loader(path_to_dataset='/home/ifeeding99/Downloads/brain-tumor-dataset/Testing'
                                             , batch_size=1, t_train=False, shuffle=False)
    
    model = ViTRM(n_blocks=1,embed_dim=100,n_heads=10, patch_size=10,
                  M = 5, T = 1, n_classes=4, input_img_size=300)
    
    train_ViTRM(model, train_loader, val_loader, n_epochs=10, 
                num_classes=4, n_supevision_steps=5, lr=1e-3, threshold_tau=0.5)
    
