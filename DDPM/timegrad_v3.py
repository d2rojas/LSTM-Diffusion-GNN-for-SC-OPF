import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os

class TimeGradDiffusion(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, n_steps=1000, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_steps = n_steps
        
        # Diffusion parameters
        self.beta = torch.linspace(beta_start, beta_end, n_steps)
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Context encoder for weather features
        self.context_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim),  # 3 weather features: temperature, wind speed, radiation
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # LSTM for temporal dependencies
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            dropout=0.1,
            batch_first=True
        )
        
        # Time-aware attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=0.1
        )
        
        # Noise prediction network with context
        self.noise_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),  # *3 for concatenated features (LSTM + attention + context)
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, x, t, context=None, hidden=None):
        # Time embedding
        t_emb = t.float().view(-1, 1) / self.n_steps
        t_emb = self.time_embed(t_emb)
        
        # Context encoding
        if context is not None:
            # context: (batch, seq, features)
            batch_size, seq_len, feat_dim = context.shape
            context_flat = context.view(-1, feat_dim)  # (batch*seq, features)
            context_emb_flat = self.context_encoder(context_flat)  # (batch*seq, hidden_dim)
            context_emb = context_emb_flat.view(batch_size, seq_len, self.hidden_dim)  # (batch, seq, hidden_dim)
        else:
            context_emb = torch.zeros(x.size(0), x.size(1), self.hidden_dim, device=x.device)
        
        # LSTM processing
        lstm_out, (h_n, c_n) = self.lstm(x, hidden)
        
        # Self-attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Combine LSTM and attention outputs
        combined = lstm_out + attn_out
        
        # Concatenate with time embedding and context
        t_emb_expanded = t_emb.unsqueeze(1).expand(-1, combined.size(1), -1)
        features = torch.cat([combined, t_emb_expanded, context_emb], dim=-1)
        
        # Predict noise
        predicted_noise = self.noise_predictor(features)
        
        return predicted_noise, (h_n, c_n)
    
    def add_noise(self, x, t):
        """Add noise to the input according to the diffusion process."""
        noise = torch.randn_like(x)
        alpha_bar_t = self.alpha_bar[t].view(-1, 1, 1)
        noisy_x = torch.sqrt(alpha_bar_t) * x + torch.sqrt(1 - alpha_bar_t) * noise
        return noisy_x, noise
    
    def remove_noise(self, x, t, context=None, hidden=None):
        """Remove noise from the input using the learned reverse process."""
        predicted_noise, new_hidden = self(x, t, context, hidden)
        alpha_t = self.alpha[t].view(-1, 1, 1)
        alpha_bar_t = self.alpha_bar[t].view(-1, 1, 1)
        beta_t = self.beta[t].view(-1, 1, 1)
        
        # Denoising formula
        mean = (1 / torch.sqrt(alpha_t)) * (x - (beta_t / torch.sqrt(1 - alpha_bar_t)) * predicted_noise)
        
        # Check if t is a tensor and handle accordingly
        if isinstance(t, torch.Tensor):
            noise = torch.randn_like(x)
            variance = torch.sqrt(beta_t) * noise
            # Only add variance for non-zero timesteps
            mask = (t > 0).view(-1, 1, 1)
            variance = variance * mask
        else:
            if t > 0:
                noise = torch.randn_like(x)
                variance = torch.sqrt(beta_t) * noise
            else:
                variance = torch.zeros_like(x)
            
        return mean + variance, new_hidden
    
    def sample(self, n_samples, seq_length, context=None, device='cuda'):
        """Generate samples using the reverse diffusion process with conditional context."""
        self.eval()
        with torch.no_grad():
            # Start from pure noise
            x = torch.randn(n_samples, seq_length, self.input_dim).to(device)
            hidden = None
            
            # Gradually denoise
            for t in tqdm(reversed(range(self.n_steps)), desc="Sampling"):
                t_batch = torch.ones(n_samples, dtype=torch.long).to(device) * t
                x, hidden = self.remove_noise(x, t_batch, context, hidden)
            
            return x

class PVDataset(Dataset):
    def __init__(self, data, weather_data, seq_length=24):
        self.data = torch.FloatTensor(data)
        self.weather_data = torch.FloatTensor(weather_data)
        self.seq_length = seq_length
        
    def __len__(self):
        return len(self.data) - self.seq_length
        
    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_length]
        y = self.data[idx + 1:idx + self.seq_length + 1]
        context = self.weather_data[idx:idx + self.seq_length]
        return x, y, context

def load_and_preprocess_data(file_path, seq_length=24):
    # Load data
    df = pd.read_csv(file_path, skiprows=10)
    
    # Convert relevant columns to numeric, coercing errors
    for col in ['P', 'G(i)', 'T2m', 'WS10m']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop rows with NaNs in any of the used columns
    df = df.dropna(subset=['P', 'G(i)', 'T2m', 'WS10m'])
    
    # Extract features
    features = ['temperature', 'humidity', 'wind_speed', 'cloud_cover', 'radiation', 'pv_generation']
    weather_features = ['T2m', 'WS10m', 'G(i)']
    
    # Create PV data from available features
    pv_data = np.zeros((len(df), len(features)))
    pv_data[:, 0] = df['T2m'].values  # temperature
    pv_data[:, 1] = 50.0  # humidity (default value as it's not in the data)
    pv_data[:, 2] = df['WS10m'].values  # wind_speed
    pv_data[:, 3] = 50.0  # cloud_cover (default value as it's not in the data)
    pv_data[:, 4] = df['G(i)'].values  # radiation
    pv_data[:, 5] = df['P'].values / 1000.0  # pv_generation (convert from W to kW)
    
    # Normalize PV data
    pv_scaler = StandardScaler()
    pv_data_normalized = pv_scaler.fit_transform(pv_data)
    
    # Prepare weather data
    weather_data = df[weather_features].values
    weather_scaler = StandardScaler()
    weather_data_normalized = weather_scaler.fit_transform(weather_data)
    
    # Create dataset
    dataset = PVDataset(pv_data_normalized, weather_data_normalized, seq_length)
    
    return dataset, pv_scaler, weather_scaler

def train_timegrad_diffusion(model, train_loader, val_loader, n_epochs=100, lr=1e-4, device='cuda'):
    """Train the TimeGrad diffusion model."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(n_epochs):
        # Training
        model.train()
        train_loss = 0
        for batch_x, batch_y, batch_context in tqdm(train_loader, desc=f'Epoch {epoch+1}/{n_epochs}'):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_context = batch_context.to(device)
            
            # Sample random timestep
            t = torch.randint(0, model.n_steps, (batch_x.shape[0],), device=device)
            
            # Add noise to target
            noisy_y, noise = model.add_noise(batch_y, t)
            
            # Predict noise
            predicted_noise, _ = model(noisy_y, t, batch_context)
            
            # Calculate loss
            loss = F.mse_loss(predicted_noise, noise)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y, batch_context in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                batch_context = batch_context.to(device)
                t = torch.randint(0, model.n_steps, (batch_x.shape[0],), device=device)
                noisy_y, noise = model.add_noise(batch_y, t)
                predicted_noise, _ = model(noisy_y, t, batch_context)
                val_loss += F.mse_loss(predicted_noise, noise).item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        scheduler.step()
        
        print(f'Epoch {epoch+1}/{n_epochs}:')
        print(f'Training Loss: {train_loss:.6f}')
        print(f'Validation Loss: {val_loss:.6f}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_timegrad_diffusion.pth')
    
    return train_losses, val_losses

def plot_results(real_data, generated_data, scaler):
    """Plot comparison between real and generated data."""
    # Inverse transform the data
    real_data = scaler.inverse_transform(real_data.reshape(-1, real_data.shape[-1]))
    generated_data = scaler.inverse_transform(generated_data.reshape(-1, generated_data.shape[-1]))
    
    # Calculate and print the mean of the generated distribution
    mean_generated = np.mean(generated_data[:, -1])
    print(f"Mean of Generated PV Generation: {mean_generated:.2f} kW")
    
    # Plot distribution
    plt.figure(figsize=(10, 6))
    plt.hist(real_data[:, -1], bins=50, alpha=0.5, label='Real PV Generation')
    plt.hist(generated_data[:, -1], bins=50, alpha=0.5, label='Generated PV Generation')
    plt.xlabel('PV Generation')
    plt.ylabel('Frequency')
    plt.title('Distribution of Real vs Generated PV Generation')
    plt.legend()
    plt.savefig('results_timegrad/timegrad_v3_distribution.png')
    plt.close()
    
    # Plot time series
    plt.figure(figsize=(15, 6))
    plt.plot(real_data[:24, -1], label='Real PV Generation', alpha=0.7)
    plt.plot(generated_data[:24, -1], label='Generated PV Generation', alpha=0.7)
    plt.title('PV Generation Over Time (First 24 Hours)')
    plt.xlabel('Hour')
    plt.ylabel('PV Generation')
    plt.legend()
    plt.savefig('results_timegrad/timegrad_v3_timeseries.png')
    plt.close()

def plot_sampling_process(model, context, n_samples=5, seq_length=24, device='cuda', scaler=None):
    """Plot the sampling process at different timesteps."""
    model.eval()
    with torch.no_grad():
        # Start from pure noise
        x = torch.randn(n_samples, seq_length, model.input_dim).to(device)
        hidden = None
        
        # Select timesteps to visualize (e.g., every 200 steps)
        timesteps_to_plot = [0, 200, 400, 600, 800, 999]
        samples_at_timesteps = []
        
        # Gradually denoise and store samples at selected timesteps
        for t in tqdm(reversed(range(model.n_steps)), desc="Sampling"):
            t_batch = torch.ones(n_samples, dtype=torch.long).to(device) * t
            x, hidden = model.remove_noise(x, t_batch, context, hidden)
            
            if t in timesteps_to_plot:
                samples_at_timesteps.append(x.cpu().numpy())
        
        # Plot the evolution of samples
        plt.figure(figsize=(15, 10))
        for i, t in enumerate(timesteps_to_plot):
            # Get the samples at this timestep
            samples = samples_at_timesteps[i]
            
            # Inverse transform the data
            if scaler is not None:
                samples = scaler.inverse_transform(samples.reshape(-1, samples.shape[-1]))
                samples = samples.reshape(n_samples, seq_length, -1)
            
            # Plot each sample
            for j in range(n_samples):
                plt.subplot(len(timesteps_to_plot), n_samples, i * n_samples + j + 1)
                plt.plot(samples[j, :, -1])  # Plot PV generation
                plt.title(f't={t}, Sample {j+1}')
                plt.xticks([])
                plt.yticks([])
        
        plt.tight_layout()
        plt.savefig('results_timegrad/timegrad_v3_sampling_process.png')
        plt.close()

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Model parameters
    input_dim = 6  # temperature, humidity, wind_speed, cloud_cover, radiation, pv_generation
    hidden_dim = 256
    n_steps = 1000
    seq_length = 24  # 24-hour window
    
    # Create results directory
    os.makedirs('results_timegrad', exist_ok=True)
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    dataset, pv_scaler, weather_scaler = load_and_preprocess_data('data/Timeseries_32.881_-117.233_E5_84kWp_crystSi_14_31deg_3deg_2022_2023.csv', seq_length)
    
    # Split data into train and validation sets
    train_size = int(0.8 * len(dataset))
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Initialize model
    print("Initializing model...")
    model = TimeGradDiffusion(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        n_steps=n_steps
    ).to(device)
    
    # Train model
    print("Starting training...")
    train_losses, val_losses = train_timegrad_diffusion(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=100,
        lr=1e-4,
        device=device
    )
    
    # Plot training curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.savefig('results_timegrad/timegrad_v3_training_curves.png')
    plt.close()
    
    # Generate samples with weather context
    print("Generating samples...")
    model.load_state_dict(torch.load('best_timegrad_diffusion.pth'))
    
    # Get weather context from validation set and repeat it to match the number of samples
    val_batch = next(iter(val_loader))
    context = val_batch[2][:32].to(device)  # Get first 32 samples
    context = context.repeat(4, 1, 1)  # Repeat to get 100 samples (32 * 4 = 128, we'll use first 100)
    context = context[:100]  # Take first 100 samples
    
    # Plot the sampling process
    print("Plotting sampling process...")
    plot_sampling_process(model, context[:5], n_samples=5, seq_length=seq_length, device=device, scaler=pv_scaler)
    
    # Generate final samples with the weather context
    samples = model.sample(n_samples=100, seq_length=seq_length, context=context, device=device)
    
    # Get some real data for comparison
    real_data = val_batch[1][:100].cpu().numpy()
    
    # Plot results
    plot_results(real_data, samples.cpu().numpy(), pv_scaler)
    print("Results have been saved in the results_timegrad directory.")

if __name__ == "__main__":
    main() 