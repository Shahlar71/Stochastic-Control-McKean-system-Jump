import matplotlib.pyplot as plt
import numpy as np

# Initialize lists to store losses and values of X_t and Y_t
losses = []
x_values = []
y_values = []

# Training loop
def train(z_net, steps=11000, lr=1e-3, M=40, batch_size=1024, device='cpu', x0=1.0, y0=0.0, p=0.05, b=0.25):
    y0_param = torch.nn.Parameter(torch.tensor([y0], device=device))  # Make Y_0 trainable
    z_net.to(device)
    
    for step in range(steps):
        # Simulate the forward-backward SDE system
        x, y = simulate_jump_diffusion(x0, y0_param, z_net, T=1.0, M=M, batch_size=batch_size, p=p, b=b, device=device)

        # Compute the loss
        loss = loss_function(y, x)

        # Append values to track loss, X_t and Y_t
        losses.append(loss.item())
        x_values.append(x.mean().item())  # Track the mean of X_t
        y_values.append(y.mean().item())  # Track the mean of Y_t
        
        # Gradient descent step
        z_net, y0_param = gradient_step(z_net, y0_param, loss, lr=lr)

        # Print progress every 500 steps
        if step % 500 == 0:
            print(f"Step {step}, Loss: {loss.item():.6f}")

    return z_net, y0_param

# Initialize the neural network model
z_net = ZNet(input_dim=2, hidden_dim=10).to('cuda' if torch.cuda.is_available() else 'cpu')

# Run training
z_net_trained, y0_trained = train(z_net, steps=11000, lr=1e-3, M=40, batch_size=1024, device=z_net.device)

# Plotting the results
plt.figure(figsize=(12, 6))

# Subplot for Loss Over Time
plt.subplot(1, 2, 1)

plt.plot(np.arange(0, len(losses) * 500, 500), losses)
plt.title("Training Loss Over Time")
plt.xlabel("Steps")
plt.ylabel("Loss")

# Subplot for X_t and Y_t Evolution
plt.subplot(1, 2, 2)
plt.plot(np.arange(0, len(x_values) * 500, 500), x_values, label="X_t")
plt.plot(np.arange(0, len(y_values) * 500, 500), y_values, label="Y_t")
plt.title("Evolution of X_t and Y_t Over Time")
plt.xlabel("Steps")
plt.ylabel("Values")
plt.legend()

# Display the plots
plt.tight_layout()
plt.show()
