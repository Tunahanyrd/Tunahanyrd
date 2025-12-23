from sklearn.metrics import classification_report
import torch

def predict(model, clf, sbert, title):
    sbert_emb = sbert.encode([title], normalize_embeddings=True)
    sbert_emb = torch.tensor(sbert_emb, dtype=torch.float).to("cuda")

    edge_index = torch.tensor([[0], [0]], dtype=torch.long).to("cuda")
    edge_attr  = torch.tensor([[1.0]], dtype=torch.float).to("cuda")  # Kendisiyle benzerliği 1.0

    with torch.no_grad():
        out = model(sbert_emb, edge_index, edge_attr)

    pred = clf.predict(out.cpu().numpy())
    return int(pred[0])

def predict_fake(model, clf, sbert, title):
    label = predict(model, clf, sbert, title)
    if label == 1:
        return (f"fake")
    else:
        return (f"True")


def predict_clf(model, data):
    model.eval()
    with torch.no_grad():
        out = model(data.x)
        preds = out.argmax(dim=1)
        print(classification_report(data.y.cpu(), preds.cpu()))