import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt

# -----------------------
# Config
# -----------------------
CSV_PATH = "xy_data.csv"
T_MIN, T_MAX = 6.0, 60.0

# bounds
BOUNDS = [
    (np.deg2rad(0.0 + 1e-3), np.deg2rad(50.0 - 1e-3)),
    (-0.05 + 1e-3, 0.05 - 1e-3),
    (0.0 + 1e-3, 100.0 - 1e-3),
]

WARM_START = np.array([0.492161, 0.021583, 54.916793])
np.random.seed(42)

# -----------------------
# Load data
# -----------------------
df = pd.read_csv(CSV_PATH)
x_obs = df["x"].to_numpy(float)
y_obs = df["y"].to_numpy(float)
n = len(df)
t_vals = np.linspace(T_MIN, T_MAX, n)

# -----------------------
# Model
# -----------------------
def predict(params, t):
    theta, M, X = params
    base = np.exp(M * np.abs(t)) * np.sin(0.3 * t)
    x = t * np.cos(theta) - base * np.sin(theta) + X
    y = 42.0 + t * np.sin(theta) + base * np.cos(theta)
    return x, y

# -----------------------
# Loss
# -----------------------
def objective(params):
    x_p, y_p = predict(params, t_vals)
    return np.mean(np.abs(x_p - x_obs) + np.abs(y_p - y_obs))

# -----------------------
# Optimize
# -----------------------
print("Running optimization...")

res_de = differential_evolution(objective, BOUNDS, seed=42, maxiter=200, tol=1e-6, polish=False)
res_local = minimize(objective, res_de.x, method="L-BFGS-B", bounds=BOUNDS, options={"maxiter":2000})

try:
    res_ws = minimize(objective, WARM_START, method="L-BFGS-B", bounds=BOUNDS, options={"maxiter":2000})
    if res_ws.fun < res_local.fun:
        res_local = res_ws
except:
    pass

theta_opt, M_opt, X_opt = res_local.x
final_cost = objective(res_local.x)

print("\n✅ Optimization finished")
print(f"Theta  = {np.rad2deg(theta_opt):.6f}°")
print(f"M      = {M_opt:.6f}")
print(f"X      = {X_opt:.6f}")
print(f"L1 err = {final_cost:.6f}")

# -----------------------
# Prediction
# -----------------------
x_pred, y_pred = predict([theta_opt, M_opt, X_opt], t_vals)

# -----------------------
# Nearest neighbor error (for lines)
# -----------------------
pred_pts = np.column_stack([x_pred, y_pred])
data_pts = np.column_stack([x_obs, y_obs])
tree = cKDTree(pred_pts)
dists, idx = tree.query(data_pts, k=1)
nearest = pred_pts[idx]

# -----------------------
# ✅ YOUR PLOT (save it)
# -----------------------
plt.figure(figsize=(7,7))
plt.scatter(x_obs, y_obs, s=18, label="Actual points", color="black")
plt.plot(x_pred, y_pred, lw=2.2, label="Predicted curve", color="blue")

# red error lines
for (ax, ay), (px, py) in zip(data_pts, nearest):
    plt.plot([ax, px], [ay, py], "--", lw=0.8, color="red", alpha=0.5)

plt.axis("equal")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Actual vs Predicted Curve")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("actual_vs_predicted.png", dpi=300, bbox_inches='tight')  # << SAVED HERE
plt.show()

print("\n📁 Saved plot as: actual_vs_predicted.png")
