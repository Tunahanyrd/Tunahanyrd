import torch
from tqdm import tqdm
def train_model(model, data, get_triplets, triplet_loss, optimizer, scheduler=None, epochs=15, patience=2):
    best_loss = float("inf")
    counter=0
    
    for epoch in tqdm(range(epochs)):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr)
        triplets = get_triplets(out.detach().cpu().numpy())
        loss = triplet_loss(out, triplets)
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step(epoch + counter)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

        if loss.item() < best_loss - 1e-4:
            best_loss = loss.item()
            counter = 0
            torch.save(model.state_dict(), "best_model.pt")
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                model.load_state_dict(torch.load("best_model.pt"), weights_only=True)
                break

    return model

def train_supervised_model(model, data, criterion, optimizer, scheduler=None, epochs=15, patience=2):
    best_loss = float("inf")
    counter=0
    
    for epoch in tqdm(range(epochs)):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step(epoch + counter)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

        if loss.item() < best_loss - 1e-4:
            best_loss = loss.item()
            counter = 0
            torch.save(model.state_dict(), "best_model.pt")
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                model.load_state_dict(torch.load("best_model.pt"))
                break

    return model

def train_mlp_model(model, x, y, criterion, optimizer, scheduler=None, epochs=60, patience=5):
    best_loss = float("inf")
    counter = 0

    for epoch in tqdm(range(epochs)):
        model.train()
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step(epoch + counter)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

        if loss.item() < best_loss - 1e-4:
            best_loss = loss.item()
            counter = 0
            torch.save(model.state_dict(), "best_mlp.pt")
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                model.load_state_dict(torch.load("best_mlp.pt"))
                break

    return model


















