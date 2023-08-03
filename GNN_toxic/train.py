import torch 
from torch_geometric.data import DataLoader
from sklearn.metrics import confusion_matrix, f1_score, \
    accuracy_score, precision_score, recall_score, roc_auc_score
import numpy as np
from tqdm import tqdm
from dataset import MoleculeDataset
from gnn_model import GNN
import mlflow.pytorch
import matplotlib.pyplot as plt 
import seaborn as sns
import pandas as pd 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Device: {device}')

def train_one_epoch(epoch, model, train_loader, optimizer, loss_fn):
    # Enumerate over the data
    all_preds = []
    all_labels = []
    for _, batch in enumerate(tqdm(train_loader)):
        # Use device
        batch.to(device)  
        # Reset gradients
        optimizer.zero_grad() 
        # Passing the node features and the connection info
        pred = model(batch.x.float(), 
                                batch.edge_attr.float(),
                                batch.edge_index, 
                                batch.batch) 
        # Calculating the loss and gradients
        loss = torch.sqrt(loss_fn(pred, batch.y))    
        loss.backward()  
        # Update using the gradients
        optimizer.step()  

        all_preds.append(np.argmax(pred.cpu().detach().numpy(), axis=1))
        all_labels.append(batch.y.cpu().detach().numpy())
    all_preds = np.concatenate(all_preds).ravel()
    all_labels = np.concatenate(all_labels).ravel()
    calculate_metrics(all_preds, all_labels, epoch, "train")
    return loss

def test(epoch, model, test_loader, loss_fn):
    all_preds = []
    all_labels = []
    for batch in test_loader:
        batch.to(device)  
        pred = model(batch.x.float(), 
                        batch.edge_attr.float(),
                        batch.edge_index, 
                        batch.batch) 
        loss = torch.sqrt(loss_fn(pred, batch.y))    
        all_preds.append(np.argmax(pred.cpu().detach().numpy(), axis=1))
        all_labels.append(batch.y.cpu().detach().numpy())
    
    all_preds = np.concatenate(all_preds).ravel()
    all_labels = np.concatenate(all_labels).ravel()
    calculate_metrics(all_preds, all_labels, epoch, "test")
    return loss

def calculate_metrics(y_pred, y_true, epoch, type):
    print(f"\n Confusion matrix: \n {confusion_matrix(y_pred, y_true)}")
    print(f"F1 Score: {f1_score(y_pred, y_true)}")
    print(f"Accuracy: {accuracy_score(y_pred, y_true)}")
    print(f"Precision: {precision_score(y_pred, y_true)}")
    print(f"Recall: {recall_score(y_pred, y_true)}")

def train(epochs):
    # Loading the dataset
    print("Loading dataset...")
    train_dataset = MoleculeDataset(root="./data/", filename="train.csv")
    test_dataset = MoleculeDataset(root="./data/", filename="test.csv", test=True)

    print(f'Length of the train dataset is {len(train_dataset)}')
    print(f'Length of the test dataset is {len(test_dataset)}')

    # Prepare training
    NUM_GRAPHS_PER_BATCH = 256
    train_loader = DataLoader(train_dataset, batch_size=NUM_GRAPHS_PER_BATCH, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=NUM_GRAPHS_PER_BATCH, shuffle=True)

    # Loading the model
    model = GNN(feature_size=train_dataset[0].x.shape[1])
    print(model)

    # Loss and Optimizer
    weights = torch.tensor([1, 10], dtype=torch.float32).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)  
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
            

    # Start training
    best_loss = np.inf
    for epoch in range(epochs): 
        # Training
        model.train()
        loss = train_one_epoch(epoch, model, train_loader, optimizer, loss_fn)
        print(f"Epoch {epoch} | Train Loss {loss}")

        # Testing
        model.eval()
        if epoch % 5 == 0:
            loss = test(epoch, model, test_loader, loss_fn)
            print(f"Epoch {epoch} | Test Loss {loss}")
                
            # Update best loss
            if float(loss) < best_loss:
                best_loss = loss

            scheduler.step()

    print(f"Finishing training with best test loss: {best_loss}")
    return model, best_loss

