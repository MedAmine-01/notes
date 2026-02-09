import numpy as np
import matplotlib.pyplot as plt


f = 50.0                         
Ts = 416e-6                      
duration = 0.1                 
loss_rate = 0.40              
freq_err = 0.02

t = np.arange(0, duration, Ts)
y_true = 400000 * np.sin(2 * np.pi * (f+freq_err) * t)


rng = np.random.default_rng(42) 
mask = rng.random(len(t)) > loss_rate


Delta = 2 * np.pi * f * Ts
c = 2 * np.cos(Delta)

# This will hold ONLY the points we actually reconstructed
y_rec_only = np.full_like(y_true, np.nan)

# Initialization
valid_idx = np.where(mask)[0]
y_prev = y_true[valid_idx[0]]
y_curr = y_true[valid_idx[1]]

print(f"{'Index':<8} | {'True Value':<12} | {'Predicted':<12} | {'Difference':<12}")
print("-" * 55)

# 5. Reconstruction Loop
for i in range(valid_idx[1] + 1, len(t)):
    # Calculate the prediction (the oscillator "flywheel")
    y_next_pred = c * y_curr - y_prev
    
    if not mask[i]:
        # PACKET LOST: use the prediction and print the difference
        y_rec_only[i] = y_next_pred
        y_actual_to_use = y_next_pred
        
        difference = y_true[i] - y_next_pred
        print(f"{i:<8} | {y_true[i]:<12.2f} | {y_next_pred:<12.2f} | {difference:<12.2f}")
    else:
        # PACKET RECEIVED: use the true value
        y_actual_to_use = y_true[i]
    
    # Update states for next iteration
    y_prev, y_curr = y_curr, y_actual_to_use

# 6. Plotting
plt.figure(figsize=(12, 6))

# Plot True Samples as background
plt.scatter(t, y_true, color='gray', s=10, label='True Signal Path', alpha=0.3)

# Plot ONLY Reconstructed points (Big hollow circles)
# Note: we only plot where we actually had to predict (mask is False)
plt.scatter(t[~mask], y_rec_only[~mask], color='black', s=5, facecolors='none', 
            edgecolors='black', label='Oscillator Prediction (Big)', zorder=5)

# Plot the Actual Lost values (Small red dots inside the big circles)
plt.scatter(t[~mask], y_true[~mask], color='red', s=20, label='Actual Value (Small)', facecolors='none', edgecolors='red', linewidth=1.5)

plt.title(f"Packet Loss Reconstruction (f={f}Hz, Loss={loss_rate*100}%)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)

plt.show()