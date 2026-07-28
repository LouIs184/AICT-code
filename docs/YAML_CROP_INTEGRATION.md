# 投影域黑边裁剪：YAML 配置化改造方案

## 一、改造目标

在 YAML 配置文件中加一个 `crop_rows` 字段，控制投影域训练/测试时上下裁剪的行数，使得：

```
proj_reader 输出: (128, 512)  →  裁剪 crop_rows 行上下 →  (112, 512)  送入模型
                                        ↑
                              YAML: data.crop_rows = 8
```

**改动范围**：4 个文件 + 1 个 YAML

---

## 二、数据流全景

```
YAML 配置                         训练/测试脚本
┌─────────────────┐              ┌──────────────────┐
│ data:            │  ──load──►   │ Trainer           │
│   reader: proj   │              │  _build_dataloader │
│   norm_clip_max  │              │    └─► proj_reader │
│   crop_rows: 8   │              │         ├── __getitem__: 读视角→裁剪→归一化
│   ...            │              │         └── 返回 (ldct, rdct): (1,112,512)
└─────────────────┘              │                   │
                                  │  model(ldct)      │
                                  │  loss(pred, rdct) │
                                  └──────────────────┘
                                        ↑
                                test.py ──┘
                                _infer_proj(): 同裁剪逻辑
```

---

## 三、具体改动

### 3.1 `src/config.py` — DataConfig 新增字段

当前 `DataConfig`：

```python
@dataclass
class DataConfig:
    reader: str = "raw"
    train_txt: str = "./train_img.txt"
    valid_txt: str = "./valid_img.txt"
    batch_size: int = 12
    valid_batch_size: int = 1
    num_workers: int = 8
    norm_clip_max: float = 0.28    # used by proj_reader
    shuffle: bool = True
```

✅ **新增 `crop_rows`**：

```python
@dataclass
class DataConfig:
    reader: str = "raw"
    train_txt: str = "./train_img.txt"
    valid_txt: str = "./valid_img.txt"
    batch_size: int = 12
    valid_batch_size: int = 1
    num_workers: int = 8
    norm_clip_max: float = 0.28    # used by proj_reader
    crop_rows: int = 0             # 投影域裁剪行数（上下各裁 N 行，0=不裁剪）
    shuffle: bool = True
```

- `crop_rows=0`（默认）：行为不变，不裁剪
- `crop_rows=8`：每个视角从 (128, 512) 裁为 (112, 512)
- `crop_rows=10`：裁为 (108, 512) — 注意这需要配合 padding 方案

> ✳️ 如果未来要用非 16 倍数的高度，还需要在 DataConfig 中加 `crop_pad` 相关字段。当前先只做 `crop_rows`，假设裁剪后的尺寸（128 - 2×crop_rows）**必须是 16 的倍数**。

### 3.2 `src/data/proj_reader.py` — 添加裁剪逻辑

**改动**：`__init__` 接收 `crop_rows`，`__getitem__` 在返回前裁剪。

```diff
 class proj_reader(Dataset):
-    def __init__(self, paired_img_txt, norm_clip_max=0.28):
+    def __init__(self, paired_img_txt, norm_clip_max=0.28, crop_rows=0):
         super().__init__()
         try:
             with open(paired_img_txt, encoding="utf-8") as f:
                 self.paired_files = f.readlines()
         except UnicodeDecodeError:
             with open(paired_img_txt, encoding=locale.getpreferredencoding()) as f:
                 self.paired_files = f.readlines()
         self.norm_clip_max = norm_clip_max
+        self.crop_rows = crop_rows
 
     def __len__(self):
         return len(self.paired_files)
 
     def __getitem__(self, index):
         line = self.paired_files[index].strip()
         input_path, label_path, view_idx = line.split()
         view_idx = int(view_idx)
 
         # Read only the target view (128×512) via file seek
         view_size = 128 * 512
         byte_offset = view_idx * view_size * 4  # float32 = 4 bytes
 
         input_view = np.fromfile(
             input_path, dtype=np.float32,
             count=view_size, offset=byte_offset
         ).reshape(128, 512)
 
         label_view = np.fromfile(
             label_path, dtype=np.float32,
             count=view_size, offset=byte_offset
         ).reshape(128, 512)
 
+        # ═══ 新增：上下裁剪黑边 ═══
+        if self.crop_rows > 0:
+            input_view = input_view[self.crop_rows:-self.crop_rows, :]  # (H-2N, 512)
+            label_view = label_view[self.crop_rows:-self.crop_rows, :]
+        # ════════════════════════════
+
         # Normalize to [0, 1]
         input_view = np.clip(input_view, 0, self.norm_clip_max) / self.norm_clip_max
         label_view = np.clip(label_view, 0, self.norm_clip_max) / self.norm_clip_max
 
         return {
             "ldct": torch.FloatTensor(input_view[np.newaxis, ...]),
             "rdct": torch.FloatTensor(label_view[np.newaxis, ...]),
         }
```

**效果**：
- `crop_rows=0` → 返回 `(1, 128, 512)`（不裁剪，向后兼容）
- `crop_rows=8` → 返回 `(1, 112, 512)`（裁剪后）
- 裁剪在**归一化之前**进行。原因是黑边像素值接近 0，clip 后也是 0，先裁再归一化结果一样，但计算量更小。

### 3.3 `src/trainer.py` — 传递 crop_rows 给 proj_reader

```diff
     def _build_dataloaders(self):
         dc = self.cfg.data
         reader_cls = READER_MAP.get(dc.reader)
         if reader_cls is None:
             raise ValueError(f"Unknown reader: {dc.reader}")
 
         reader_kwargs = {}
         if dc.reader == "proj":
             reader_kwargs["norm_clip_max"] = dc.norm_clip_max
+            if dc.crop_rows > 0:
+                reader_kwargs["crop_rows"] = dc.crop_rows
 
         self.train_dataset = reader_cls(paired_img_txt=dc.train_txt, **reader_kwargs)
         self.valid_dataset = reader_cls(paired_img_txt=dc.valid_txt, **reader_kwargs)
```

或者更简洁，因为 `crop_rows=0` 是 proj_reader 的默认值，可以直接传递：

```python
reader_kwargs["crop_rows"] = dc.crop_rows  # 总是传递，0=不裁剪
```

这样更干净。

### 3.4 `test.py` — 测试脚本也支持裁剪

测试时同样需要裁剪 input 和 gt。但 `test.py` 目前没有用 Reader 类，是自己直接读 raw 的。

**方案**：从 YAML 配置中读取 `crop_rows`，在推理循环中做相同裁剪。

```diff
 def _infer_proj(args, model):
     """Projection-domain inference (ProjCT): process (720, 128, 512) volumes."""
     default_patients = [...]
     patients = args.patient or default_patients
 
+    # 从 config 读取 crop_rows（默认 0）
+    infer_cfg = load_config(args.config)
+    crop_rows = getattr(infer_cfg.data, "crop_rows", 0)
+    # 推理时目标尺寸 = 128 - 2*crop_rows
+    target_rows = 128 - 2 * crop_rows if crop_rows > 0 else 128
+
     for patient in patients:
         print(f"\nTesting {patient} (proj domain) ...")
         patient_dir = os.path.join(args.data_dir, patient)
 
         vol = np.fromfile(
             os.path.join(patient_dir, "proj_pbi_fs_dec_blur_phase_100000.raw"),
             dtype=np.float32,
         ).reshape(720, 128, 512)
 
         gt = np.fromfile(
             os.path.join(patient_dir, "proj_no_pbi.raw"),
             dtype=np.float32,
         ).reshape(720, 128, 512)
 
         norm_clip_max = 0.28
         preds = []
         for i in tqdm(range(720), desc=f"  Inferring {patient}"):
             s = vol[i]  # (128, 512)
+
+            # ═══ 新增：裁剪黑边（与训练一致）═══
+            if crop_rows > 0:
+                s = s[crop_rows:-crop_rows, :]
             s = (np.clip(s, 0, norm_clip_max) / norm_clip_max).astype(np.float32)
+
             t = torch.FloatTensor(s[np.newaxis, np.newaxis, ...]).cuda()
             with torch.no_grad():
                 p = model(t)
             pred = np.squeeze(p.cpu().numpy()) * norm_clip_max
             preds.append(pred)
 
         pred_vol = np.array(preds)  # (720, target_rows, 512)
         pred_vol.tofile(os.path.join(args.output, f"{patient}_pred.raw"))
         print(f"  Saved to {args.output}/{patient}_pred.raw")
```

输出文件形状从 `(720, 128, 512)` 变为 `(720, 112, 512)`（如果 crop_rows=8）。

### 3.5 `experiments/denseunet_projct.yaml` — 加上配置

```yaml
# DenseUNet2d — ProjCT (projection domain)
experiment:
  name: DenseUNet_ProjCT
  seed: 1234

data:
  reader: proj
  train_txt: ./train_proj_img.txt
  valid_txt: ./valid_proj_img.txt
  batch_size: 36
  valid_batch_size: 1
  num_workers: 8
  shuffle: true
  norm_clip_max: 0.28
  crop_rows: 8            # ← 新增：上下各裁剪 8 行 → (112, 512)

model:
  name: DenseUNet2d
  in_chl: 1
  out_chl: 1
  model_chl: 32
  ...
```

---

## 四、verify：112 在 U-Net 中各层情况

| 层 | 操作 | 输入 H×W | 输出 H×W | 是否整数 |
|----|------|---------|---------|---------|
| begin | Conv 3×3 | 112×512 | 112×512 | ✅ |
| down1 | MaxPool/2 | 112×512 | **56**×256 | ✅ 112/2=56 |
| down2 | Conv stride=2 | 56×256 | **28**×128 | ✅ 56/2=28 |
| down3 | Conv stride=2 | 28×128 | **14**×64 | ✅ 28/2=14 |
| down4 | Conv stride=2 | 14×64 | **7**×32 | ✅ 14/2=7 |
| center | DenseBlock | 7×32 | 7×32 | ✅ |
| up4 | Upsample×2 | 7×32 | **14**×64 | ✅ 7×2=14 |
| up3 | Upsample×2 | 14×64 | **28**×128 | ✅ |
| up2 | Upsample×2 | 28×128 | **56**×256 | ✅ |
| up1 | Upsample×2 | 56×256 | **112**×512 | ✅ 与输入一致 |

全部整数，没有问题。

---

## 五、改动汇总

| 文件 | 改动类型 | 改动内容 |
|------|---------|---------|
| `src/config.py` | ✅ 新增字段 | `DataConfig.crop_rows: int = 0` |
| `src/data/proj_reader.py` | ✅ 修改 | `__init__` 接受 `crop_rows`，`__getitem__` 在归一化前裁剪 |
| `src/trainer.py` | ✅ 修改 | `_build_dataloaders()` 传递 `crop_rows` 到 `proj_reader` |
| `test.py` | ✅ 修改 | `_infer_proj()` 从 config 读取 `crop_rows` 并裁剪 |
| `experiments/denseunet_projct.yaml` | ✅ 新增配置 | `data.crop_rows: 8` |

**向后兼容性**：`crop_rows=0` 是默认值，不裁剪任何行，行为与修改前完全一致。所有现有的 PhaseCT（图像域）YAML 文件不受影响。

---

## 六、后续可能的扩展

如果以后需要裁剪到非 16 倍数的高度（比如 108），可以再加字段：

```yaml
data:
  crop_rows: 10           # 128 → 108
  crop_pad_to: 112        # 108 → pad to 112 (16的倍数)
  crop_pad_mode: replicate  # padding 模式
```

这需要 proj_reader 在裁剪后再 pad 到目标尺寸，同时在 loss 中加 mask 忽略 pad 区域。当前暂不实现。
