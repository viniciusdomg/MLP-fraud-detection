import torch
import copy

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        """
        patience: Quantas épocas esperar após a última melhoria antes de parar.
        min_delta: Diferença mínima para ser considerada uma melhoria.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model_weights = None

    def __call__(self, val_loss, model):
        # Primeira época
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model_weights = copy.deepcopy(model.state_dict())
        # Se a perda atual não melhorou em relação à melhor perda
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        # Se a perda melhorou
        else:
            self.best_loss = val_loss
            self.best_model_weights = copy.deepcopy(model.state_dict())
            self.counter = 0 # Reseta o contador