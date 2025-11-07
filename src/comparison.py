import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
import os

# ------------------------------
# Input data
# ------------------------------
stim_intensity = np.array([25, 50, 75, 100, 150, 200, 250, 300, 400, 500, 600])

# Fiber volley (FV)
fv_auto = np.array([0.014607747, 0.006103516, 0.075398763, 0.093221028, 0.23763021,
                    0.369221998, 0.446004233, 0.568969727, 0.707600896, 0.849650066,
                    0.920613608])
fv_manual = np.array([0.0565, 0.0366, 0.1312, 0.1638, 0.2706, 0.3967, 0.4496,
                      0.5809, 0.7589, 0.887, 0.9745])

# EPSP slope
epsp_auto = np.array([30.44977122, 146.2266245, 795.274635, 1115.11293, 1791.911579,
                      2288.629491, 2675.211945, 2956.027284, 3432.602521, 3730.047159,
                      3882.580671])
epsp_manual = np.array([97.0023, 123.4266, 671.7568, 1087.1041, 1861.2815,
                        2396.066, 2730.3062, 2962.3848, 3556.7515, 3853.208, 3906.2509])

# ------------------------------
# Helper: Build comparison table
# ------------------------------
def comparison_table(name, manual, auto):
    err = auto - manual
    mae = mean_absolute_error(manual, auto)
    r2 = r2_score(manual, auto)
    df = pd.DataFrame({
        "Stimulus (µA)": stim_intensity,
        f"{name}_manual": manual,
        f"{name}_auto": auto,
        f"{name}_error": err
    })
    summary = pd.DataFrame({
        "Stimulus (µA)": ["MAE", "R²"],
        f"{name}_manual": [np.nan, np.nan],
        f"{name}_auto": [np.nan, np.nan],
        f"{name}_error": [mae, r2]
    })
    return pd.concat([df, summary], ignore_index=True)

# Combine FV + EPSP
fv_df = comparison_table("fv_amp", fv_manual, fv_auto)
epsp_df = comparison_table("epsp_slope", epsp_manual, epsp_auto)
combo = pd.concat([fv_df, epsp_df.drop(columns=["Stimulus (µA)"])], axis=1)

# ------------------------------
# Save to CSV in working dir
# ------------------------------
output_path = os.path.join(os.getcwd(), "fv_epsp_comparison.csv")
combo.to_csv(output_path, index=False)
print(f"✅ Saved comparison table to: {output_path}")

# Optional: preview
pd.set_option("display.float_format", lambda x: f"{x:.4f}" if isinstance(x, float) else x)
print(combo.to_string(index=False))