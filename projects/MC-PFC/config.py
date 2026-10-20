"""MC-PFC 复现的全局配置。

所有路径都从本文件位置推导，不依赖当前工作目录。数据只读 `data/`，模型只读
`models/`，本项目的全部产物（特征缓存、日志、结果）只写入 `projects/MC-PFC/`。
"""

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_DIR.parents[1]

# ---------------------------------------------------------------------------
# 数据
# ---------------------------------------------------------------------------
DATA_DIR = REPO_ROOT / "data"
EN_ROOT = DATA_DIR / "train"          # 英文 ADReSSo（Pitt Cookie Theft）
ZH_ROOT = DATA_DIR / "test"           # 中文 NCMMSC2021 + 多录音补充数据
EN_LABELS = EN_ROOT / "adresso-train-mmse-scores.csv"
ZH_LABELS = ZH_ROOT / "test_labels.csv"

# ---------------------------------------------------------------------------
# 预训练特征提取器（论文 3.2）
# ---------------------------------------------------------------------------
HUBERT_DIR = REPO_ROOT / "models" / "hubert-large-ls960-ft"
SAMPLE_RATE = 16_000
# HuBERT 全注意力对 4 分钟长音频会爆显存；按 30 s 非重叠分块，块内取最后一层
# 隐层，最后把所有有效帧一起求均值（等价于整段录音的帧级均值）。
CHUNK_SECONDS = 30.0
FEATURE_DIM = 1024

# 特征缓存：每条录音一个 .npy，中文在建模时再做受试者级均值聚合。
FEATURE_DIR = PROJECT_DIR / "features"
LOG_DIR = PROJECT_DIR / "logs"

# ---------------------------------------------------------------------------
# MC-PFC 超参（论文 3.4 / 关键超参汇总）
# ---------------------------------------------------------------------------
N_FEATURES = 60          # ANOVA F-test top-60，只在英文上拟合
SVM_C = 10.0
SVM_GAMMA = "scale"
TAU = 0.55               # 最终决策阈值
N_EPISODES = 500         # T
K_PER_CLASS = 5          # 每轮每类抽 5 人

# 用户确认的少样本协议：固定 32 人中文标注池（16 AD + 16 HC），永久排除出测试集。
POOL_PER_CLASS = 16
POOL_SEED = 42           # 只用于固定这 32 人，与 MC 种子无关
MC_SEEDS = (0, 1, 2, 3, 4)   # 论文的 5 个独立固定种子
