import mmcv
p = "./data/nuscenes/nuscenes_infos_val_sweep.pkl"
d = mmcv.load(p)
print("info_type:", d.get("metadata",{}).get("info_type"))
print("num infos:", len(d["infos"]))
print("samples with >=1 sweep:",
      sum(1 for x in d["infos"] if x.get("sweeps") and len(x["sweeps"])>0))


import mmcv, numpy as np
d = mmcv.load("./data/nuscenes/nuscenes_infos_val_sweep.pkl")
s = d["infos"][0]
for k in ["ego2global_translation","ego2global_rotation",
          "lidar2ego_translation","lidar2ego_rotation","lidar_path"]:
    print(k, "OK" if k in s else "MISSING")
    
    