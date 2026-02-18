import torch
import matplotlib
import matplotlib.pyplot as plt
import os

matplotlib.style.use('ggplot')

class SaveBestModel:
    def __init__(self, save_path):
        self.save_path = save_path
        self.best_loss = float('inf')

    def __call__(self, model, loss):
        if loss < self.best_loss:
            self.best_loss = loss
            torch.save(model.state_dict(), self.save_path)
            print(f"New best model saved with loss: {loss:.4f}")
            os.path.join(self.save_path)

def save_model(epoch, model, optimizer, criterion, save_path, name):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'criterion_state_dict': criterion.state_dict()
    }, os.path.join(save_path, name))
    
def save_plots(train_acc,valid_acc,train_loss,valid_loss,save_path):
    #Accuracy Plots
    plt.figure(figsize=(10, 7))
    plt.subplot(1, 2, 1)
    plt.plot(train_acc, label='Train Accuracy')
    plt.plot(valid_acc, label='Validation Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(save_path, 'accuracy_plots.png'))
    plt.close()

    #Loss Plots
    plt.subplot(1, 2, 2)
    plt.plot(train_loss, label='Train Loss')
    plt.plot(valid_loss, label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'loss_plots.png'))
    plt.close()
