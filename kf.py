import numpy as np
import matplotlib.pyplot as plt


f = 50.0
Ts = 416e-6
duration = 3
loss_rate = 0.40
freq_err = 0.5
amplitude = 400000
amp2 = 10000
f2 = 150
t = np.arange(0, duration, Ts)
y_true = amplitude * np.sin(2 * np.pi * (f + freq_err) * t) + amp2 *  np.sin(2 * np.pi * f2  * t)


rng = np.random.default_rng(42)
mask = rng.random(len(t)) > loss_rate  



x = np.zeros(2)  
x[0] = y_true[0]
x[1] = np.sqrt(max(0, 400000**2 - x[0]**2)) 


cos_term = np.cos(2 * np.pi * f * Ts)
sin_term = np.sin(2 * np.pi * f * Ts)
F = np.array([[cos_term, sin_term],
              [-sin_term, cos_term]])

H = np.array([[1, 0]])  


P = np.eye(2) * 1e-6
Q = np.eye(2) * 1e-1
R = 1e-6

y_rec_only = np.full_like(y_true, np.nan)


for i in range(1, len(t)):
    x_pred = F @ x
    P_pred = F @ P @ F.T + Q

    if not mask[i]:
        
        y_next_pred = float(x_pred[0])  
        y_rec_only[i] = y_next_pred
        difference = y_true[i] - y_next_pred
        error = 100 * (difference / amplitude)
        #print(f"{i:<8} | {y_true[i]:<12.2f} | {y_next_pred:<12.2f} | {difference:<12.2f} | {error :<12.2f}")
        x = x_pred
        P = P_pred
    else:
        y_meas = y_true[i]
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T / S
        x = x_pred + (K.flatten() * (y_meas - H @ x_pred).item())  # ensure scalars
        P = (np.eye(2) - K @ H) @ P_pred


plt.figure(figsize=(12, 6))
plt.scatter(t, y_true, color='gray', s=10, label='True Signal Path', alpha=0.3)
plt.scatter(t[~mask], y_true[~mask], color='black', s=5, facecolors='none',
            edgecolors='black', label='Kalman Prediction', zorder=5)
plt.scatter(t[~mask], y_rec_only[~mask], color='red', s=20, facecolors='none',
            edgecolors='red', linewidth=1.5, label='Actual Value')
plt.title(f"Packet Loss Reconstruction with Kalman Filter (f={f}Hz, Loss={loss_rate*100}%)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()

error = y_rec_only[~mask] - y_true[~mask]
mean_error = np.mean(error)
var_error = np.var(error)
rms_error = np.sqrt(np.mean(error**2))

mean_error_percent = 100 * mean_error / amplitude
var_error_percent = 100 * var_error / (amplitude**2)
rms_error_percent = 100 * rms_error / amplitude

print(f"Mean Error: {mean_error:.2f} ({mean_error_percent:.4f} % of amplitude)")
print(f"Variance of Error: {var_error:.2f} ({var_error_percent:.6f} %^2 of amplitude)")
print(f"RMS Error: {rms_error:.2f} ({rms_error_percent:.4f} % of amplitude)")

correlation_matrix = np.corrcoef(y_true[~mask], y_rec_only[~mask])
correlation = correlation_matrix[0, 1]
print(f"Correlation coefficient (lost samples): {correlation:.6f}")