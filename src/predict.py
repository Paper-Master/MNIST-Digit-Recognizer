import torch
import pandas as pd
from src.model import Recognizer

def generate_submission(model, test_dataloader):
    model.eval()
    predictions = []

    with torch.no_grad():
        for images in test_dataloader:
            pred = model(images)
            _, predicted = torch.max(pred.data, 1)
            predictions.extend(predicted.cpu().tolist())

    submission = pd.DataFrame({
        "ImageId" : range(1, len(predictions) + 1),
        "Label" : predictions
    })

    submission.to_csv('predictions/predictions5-7-2025.csv', index=False)