import torch


def load_model(path):
    model = torch.load(path)
    return model


def predict_and_store(data_path, model_path, save_path):
    model = load_model(model_path)
    print(model)
