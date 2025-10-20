import pickle
import pprint

# Path to your pickle file
pkl_path = "./data/nuscenes/nuscenes_infos_val_sweep.pkl"

print(f"reading pickle file:{pkl_path}")

with open(pkl_path, "rb") as f:
    data = pickle.load(f)

# Print the top-level structure
print("=== Top-level keys ===")
print(data.keys())

# If it looks like {'infos': [...], 'metadata': {...}}
print("\n=== Metadata ===")
if "metadata" in data:
    pprint.pprint(data["metadata"])

print("\n=== Number of samples ===")
if "infos" in data:
    print(len(data["infos"]))

# Show one example sample info
print("\n=== First sample info (trimmed) ===")
if "infos" in data and len(data["infos"]) > 0:
    first = data["infos"][10]
    for k, v in first.items():
        if isinstance(v, (list, dict)):
            print(f"{k}: type={type(v)}, len={len(v)}")
        else:
            print(f"{k}: {v}")
