import numpy as np
import torch

class TQMseLoss(torch.nn.Module):
    def __init__(self, num_useful_qubits, num_meas_qubits):
        super().__init__()
        
        self.filter_matrix = np.zeros((num_meas_qubits, num_useful_qubits))
        
        for i in range(num_useful_qubits):
            self.filter_matrix[i, i] = 1
            
        self.filter_matrix = torch.from_numpy(self.filter_matrix).to(torch.float32)
            
    def forward(self, preds, labels):
        useful_preds = torch.matmul(preds, self.filter_matrix)
        
        loss = torch.mean(torch.sum(torch.pow(torch.sub(useful_preds, labels), 2), 1))
            
        return loss


def compute_tq_model_acc(preds, labels, num_meas_qubits):
    return torch.sum(torch.eq(torch.sum(torch.gt(torch.multiply(preds[:, :num_meas_qubits], labels), 0), 1), num_meas_qubits).to(torch.float))
 

def train_tq_model(model, num_meas_qubits, opt, loss, data_loader, test_data_loader, test_data_size, steps, print_freq=1, loss_window=50, acc_fn=None,
                  use_test=True):
    if acc_fn == None:
        acc_fn = compute_tq_model_acc
    
    losses = []
    
    for i in range(steps):
        x, y = next(iter(data_loader))
        y = y.to(torch.long)
        
        opt.zero_grad()
        
        out = model(x)
        
        batch_loss = loss(out, y)
        
        batch_loss.backward()
        
        losses.append(batch_loss.detach().item())
        
        opt.step()
        
        if not (i % print_freq) and i:
            print(f'Step {i + 1} | Loss: {(losses[-1])}')


    test_losses = []
    test_accs = []
    
    if use_test:
        for x, y in test_data_loader:
            preds = model(x)

            test_losses.append(loss(preds, y).detach().numpy())
            test_accs.append(acc_fn(preds, y, num_meas_qubits).detach().numpy())
            
        return np.mean(test_losses), np.sum(test_accs) / test_data_size
    else:
        return