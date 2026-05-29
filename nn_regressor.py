import numpy as np
import torch
import torch.nn as nn


class MLP(nn.Module):

    def __init__(self, input_dim, hidden_units=32, num_hidden_layers=2):
        super().__init__()

        layers = []
        dim_in = input_dim

        for _ in range(num_hidden_layers):
            lin = nn.Linear(dim_in, hidden_units)
            nn.init.kaiming_normal_(lin.weight, nonlinearity='relu')
            nn.init.zeros_(lin.bias)
            layers += [lin, nn.ReLU()]  # ReLU for nonlinear feature extraction
            dim_in = hidden_units

        self.hidden = nn.Sequential(*layers)
        self.out = nn.Linear(dim_in, 1)
        nn.init.kaiming_normal_(self.out.weight, nonlinearity='linear')
        nn.init.zeros_(self.out.bias)

    def forward(self, x):  # softplus output to enforce strictly positive cva estimates
        h = self.hidden(x)
        return torch.nn.functional.softplus(self.out(h))


class Regressor:
    def __init__(
        self,
        input_dim,
        hidden_units=32,
        num_hidden_layers=2,
        lr=1e-3,
        num_epochs=100,
        batch_size=256,
        val_frac=0.1,
        early_stop=True,
        patience=20,
        device='cuda',
        seed=0,
        verbose=True,
    ):
        torch.manual_seed(seed)
        self.device = torch.device(device)
        self.model = MLP(input_dim, hidden_units, num_hidden_layers).to(self.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()  # MSE loss for CVA
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.val_frac = val_frac  # data frac for validation
        self.early_stop = early_stop
        self.patience = patience
        self.verbose = verbose
        # standardization of features/labels
        # testing manual over BatchNorm first (shallow network)
        self.x_mean = None
        self.x_std = None
        self.y_std = None
        self.history = {'train_loss': [], 'val_loss': []}  # saving for diagnostic 

    def standardize_fit(self, X, y):
        """ stats for standardization where x_mean/x_std per feature
        y_std scalar for rescaling labels to order 1"""
        self.x_mean = X.mean(dim=0, keepdim=True)
        self.x_std = X.std(dim=0, keepdim=True).clamp_min(1e-7) # avoids division by zero
        self.y_std = y.std().clamp_min(1e-7)

    def to_tensor(self, a):
        if isinstance(a, np.ndarray):
            a = torch.from_numpy(a).float()
        return a.to(self.device)

    def train(self, features, labels):
        X = self.to_tensor(features)
        y = self.to_tensor(labels).view(-1, 1)

        # train/val
        # t is fixed
        n = X.shape[0]
        permut = torch.randperm(n, device=self.device)
        n_val = int(n * self.val_frac)
        idx_val, idx_tr = permut[:n_val], permut[n_val:]
        X_tr, y_tr = X[idx_tr], y[idx_tr]
        X_va, y_va = X[idx_val], y[idx_val]

        self.standardize_fit(X_tr, y_tr)
        X_tr_s = (X_tr - self.x_mean) / self.x_std
        X_va_s = (X_va - self.x_mean) / self.x_std
        y_tr_s = y_tr / self.y_std
        y_va_s = y_va / self.y_std

        best_val = float('inf')
        best_state = None
        bad_epochs = 0

        n_tr = X_tr_s.shape[0]
        for epoch in range(1, self.num_epochs + 1):
            # train
            self.model.train()
            perm_e = torch.randperm(n_tr, device=self.device)
            running = 0.0
            count = 0
            for i in range(0, n_tr, self.batch_size):
                idx = perm_e[i:i + self.batch_size]
                xb, yb = X_tr_s[idx], y_tr_s[idx]
                self.opt.zero_grad()
                pred = self.model(xb)
                loss = self.loss_fn(pred, yb)
                loss.backward()
                self.opt.step()
                running += loss.item() * xb.shape[0]
                count += xb.shape[0]
            tr_loss = running / count

            # val
            self.model.eval()
            with torch.no_grad():
                val_loss = self.loss_fn(self.model(X_va_s), y_va_s).item()

            self.history['train_loss'].append(tr_loss)
            self.history['val_loss'].append(val_loss)

            improved = val_loss < best_val - 1e-6
            if improved:
                best_val = val_loss
                best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1

            if self.verbose:
                tag = " *" if improved else ""
                print(f"epoch {epoch:3d} | train {tr_loss:.4e} | val {val_loss:.4e}{tag}")

            if self.early_stop and bad_epochs >= self.patience:
                if self.verbose:
                    print(f"early stop @ epoch {epoch} (no val improvement for {self.patience} epochs)")
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)

    def predict(self, features):
        X = self.to_tensor(features)
        X_s = (X - self.x_mean) / self.x_std
        self.model.eval()  # val mode
        with torch.no_grad():
            y_s = self.model(X_s)
        return (y_s * self.y_std).cpu().numpy().squeeze(-1)
    
    def get_state(self):
        """snapshot of NN weights at the current pricing time step (transfer learning)"""
        return {
        'model':  {k: v.detach().clone() for k, v in self.model.state_dict().items()},
        'opt': self.opt.state_dict(),
        'x_mean': self.x_mean.clone() if self.x_mean is not None else None,
        'x_std': self.x_std.clone() if self.x_std  is not None else None,
        'y_std': self.y_std.clone() if self.y_std  is not None else None}

    def set_state(self, state):
        """restoring NN weights saved at a given pricing time step"""
        self.model.load_state_dict(state['model'])
        self.opt.load_state_dict(state['opt'])
        self.x_mean = state['x_mean']
        self.x_std = state['x_std']
        self.y_std = state['y_std']