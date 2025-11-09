import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sqlalchemy import create_engine, Column, Integer, Float, String
from sqlalchemy.orm import sessionmaker, declarative_base

# ----------------------- FILE PATHS -----------------------
TRAIN_PATH = r"C:\Users\brije\Downloads\New folder\train.csv"
IDEAL_PATH = r"C:\Users\brije\Downloads\New folder\ideal.csv"
TEST_PATH  = r"C:\Users\brije\Downloads\New folder\test.csv"
OUT_DIR = os.path.dirname(TRAIN_PATH)

DB_URL  = "sqlite:///" + os.path.join(OUT_DIR, "final_data.db")
CSV_OUT = os.path.join(OUT_DIR, "test_mappings.csv")
TXT_OUT = os.path.join(OUT_DIR, "summary.txt")

PLOT_OVERVIEW = os.path.join(OUT_DIR, "svr_mapping_overview.png")
PLOT_PAIRS    = os.path.join(OUT_DIR, "svr_mapping_pairs.png")
PLOT_DEV      = os.path.join(OUT_DIR, "svr_deviation_scatter.png")

# ----------------------- LOAD DATA -----------------------
train_df = pd.read_csv(TRAIN_PATH)
ideal_df = pd.read_csv(IDEAL_PATH)
test_df  = pd.read_csv(TEST_PATH)

train_y_cols = [c for c in train_df.columns if c.startswith("y")]
ideal_y_cols = [c for c in ideal_df.columns if c.startswith("y")]

if not train_y_cols:
    raise ValueError("Training data must have columns y1..y4")

# ----------------------- DATABASE SETUP -----------------------
Base = declarative_base()

class TrainingData(Base):
    __tablename__ = "training_data"
    id = Column(Integer, primary_key=True)
    x = Column(Float)
    y1 = Column(Float)
    y2 = Column(Float)
    y3 = Column(Float)
    y4 = Column(Float)

class IdealFunctions(Base):
    __tablename__ = "ideal_functions"
    id = Column(Integer, primary_key=True)
    x = Column(Float)
    # Create all y1..y50
    locals().update({f"y{i}": Column(Float) for i in range(1, 51)})

class TestDataResults(Base):
    __tablename__ = "test_data"
    id = Column(Integer, primary_key=True)
    x = Column(Float)
    y = Column(Float)
    predicted_y = Column(Float)
    chosen_ideal_function = Column(String)
    deviation = Column(Float)

if os.path.exists(DB_URL.replace("sqlite:///", "")):
    os.remove(DB_URL.replace("sqlite:///", ""))

engine = create_engine(DB_URL)
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)

# ----------------------- TRAIN SVR MODELS -----------------------
x_train = train_df["x"].values.reshape(-1, 1)
models = {}
for y_col in train_y_cols:
    y_train = train_df[y_col].values
    svr = SVR(kernel="rbf", C=100, epsilon=0.1)
    svr.fit(x_train, y_train)
    models[y_col] = svr

# ----------------------- FIND IDEAL FUNCTIONS -----------------------
chosen = {}
mse_table = pd.DataFrame(index=train_y_cols, columns=ideal_y_cols, dtype=float)

for y_col in train_y_cols:
    x_vals = ideal_df["x"].values.reshape(-1, 1)
    pred = models[y_col].predict(x_vals)
    for ideal_col in ideal_y_cols:
        mse_table.loc[y_col, ideal_col] = mean_squared_error(ideal_df[ideal_col], pred)

    chosen[y_col] = mse_table.loc[y_col].idxmin()

# ----------------------- MAP TEST DATA -----------------------
mapped_rows = []
ideal_x = ideal_df["x"].values
sqrt2 = np.sqrt(2.0)

with Session() as session:
    for _, row in test_df.iterrows():
        xq = float(row["x"])
        yq = float(row["y"])
        xq_arr = np.array([[xq]])

        best_name = None
        best_dev = None
        best_pred = None

        for y_col, ideal_col in chosen.items():
            model = models[y_col]
            pred_y = float(model.predict(xq_arr))
            ideal_y = float(np.interp(xq, ideal_x, ideal_df[ideal_col]))
            deviation = abs(pred_y - ideal_y)

            if best_dev is None or deviation < best_dev:
                best_dev = deviation
                best_name = ideal_col
                best_pred = pred_y

        mapped_rows.append({
            "x": xq, "y": yq,
            "predicted_y": best_pred,
            "chosen_ideal_function": best_name,
            "deviation": best_dev
        })
        session.add(TestDataResults(
            x=xq, y=yq, predicted_y=best_pred,
            chosen_ideal_function=best_name, deviation=best_dev
        ))
    session.commit()

mapped_df = pd.DataFrame(mapped_rows)
mapped_df.to_csv(CSV_OUT, index=False)

# ----------------------- WRITE SUMMARY -----------------------
with open(TXT_OUT, "w", encoding="utf-8") as f:
    f.write("Chosen ideal functions (SVR-based):\n")
    for y_col in train_y_cols:
        f.write(f"  {y_col} -> {chosen[y_col]}\n")

# ----------------------- PLOTTING -----------------------
train_sorted = train_df.sort_values("x")
ideal_sorted = ideal_df.sort_values("x")
mapped_ok = mapped_df.dropna(subset=["chosen_ideal_function"])

# Overview Plot
plt.figure(figsize=(11, 6))
plt.title("SVR Predicted vs Ideal Functions and Test Mapping")
plt.xlabel("x")
plt.ylabel("y")

# Train curves
for y_col in train_y_cols:
    plt.plot(train_sorted["x"], train_sorted[y_col], linewidth=1, alpha=0.6, label=f"train {y_col}")

# Ideal curves
for ideal_col in chosen.values():
    plt.plot(ideal_sorted["x"], ideal_sorted[ideal_col], linewidth=2, alpha=0.9, label=f"ideal {ideal_col} (chosen)")

# Mapped points
if not mapped_ok.empty:
    for name, grp in mapped_ok.groupby("chosen_ideal_function"):
        plt.scatter(grp["x"], grp["y"], s=20, label=f"test→{name}")

plt.legend(loc="best")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(PLOT_OVERVIEW, dpi=150)

# Pairwise subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
axes = axes.ravel()

for i, y_col in enumerate(train_y_cols):
    ax = axes[i]
    ideal_col = chosen[y_col]
    ax.set_title(f"{y_col} vs {ideal_col}")
    ax.plot(train_sorted["x"], train_sorted[y_col], alpha=0.8, label=f"{y_col}")
    ax.plot(ideal_sorted["x"], ideal_sorted[ideal_col], alpha=0.9, label=f"{ideal_col}")
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(PLOT_PAIRS, dpi=150)

# Deviation Scatter
plt.figure(figsize=(10, 5))
plt.title("SVR Deviation Scatter")
plt.xlabel("x")
plt.ylabel("Deviation |y_pred - y_ideal|")
plt.scatter(mapped_df["x"], mapped_df["deviation"], s=18, alpha=0.9, label="Deviation")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(PLOT_DEV, dpi=150)

try:
    plt.show()
except Exception:
    print("Plot window not supported, check saved PNG files.")

# ----------------------- PRINT SUMMARY -----------------------
print("\n✅ SVR Model Completed.")
print("Selected ideal functions:")
for y_col in train_y_cols:
    print(f"  {y_col} -> {chosen[y_col]}")
print("\nOutputs:")
print(f"  Database: {DB_URL}")
print(f"  CSV:      {CSV_OUT}")
print(f"  Summary:  {TXT_OUT}")
print(f"  Plots:    {PLOT_OVERVIEW}, {PLOT_PAIRS}, {PLOT_DEV}")