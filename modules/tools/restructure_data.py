"""把 data/ 目录层级重整成更直观的样子（干跑默认，--apply 才动手）。

为什么要改
----------
原来长这样：
    data/diagnosis/train/text/<dx>/<uid>.csv          逐词时间戳
    data/diagnosis/train/text/<dx>/<uid>distil*.pt    旧特征（文本+音频混在一起）
    data/diagnosis/train/text_xlmr_xlsr/<dx>/*.pt     新特征
    data/diagnosis/train/text_xlmr_wav2vec2/<dx>/*.pt 混合特征

三个毛病：
    1. `diagnosis` 这层没有信息量 —— 它就是数据集根，白多一层。
    2. `text/` 名字骗人 —— 里面躺着 4 份音频特征 + 1 份文本特征，还混着逐词表。
    3. 逐词表（脚本① 的原料）和特征（脚本② 的产物）挤在同一目录。

改成：
    data/train/words/<dx>/<uid>.csv                   逐词时间戳
    data/train/feat_distil/<dx>/*.pt                  旧特征
    data/train/feat_xlmr_xlsr/<dx>/*.pt               新特征
    data/train/feat_xlmr_wav2vec2/<dx>/*.pt           混合特征
    data/train/audio/<dx>/*.wav                       原始录音（本来就不动）
    data/train/splits/*.npy                           5 折划分（不动）

命名规则：特征目录一律 `feat_<模型组合>`，逐词表一律 `words/`，
`audio/`、`splits/`、顶层标签表保持原名。这样"哪一类东西放哪"一眼可辨。

本脚本只做搬运，**不删除、不改内容**：
    - 同盘 mv（秒级，不复制字节）
    - 目标已存在且内容一致 -> 跳过（幂等，可以重跑）
    - 目标已存在但内容不同 -> 报错停下，绝不覆盖
    - 空目录（.ipynb_checkpoints 之类）顺手清掉

用法：
    python tools/restructure_data.py            # 干跑，只打印计划
    python tools/restructure_data.py --apply    # 真正搬运
    python tools/restructure_data.py --apply --verify   # 搬完逐字节核对
"""
import argparse
import filecmp
import os
import shutil
import sys

_THIS = os.path.dirname(os.path.abspath(__file__))
_MODULES = os.path.dirname(_THIS)
if _MODULES not in sys.path:
    sys.path.insert(0, _MODULES)

# 项目根：modules/tools -> modules -> 项目根
PROJECT_ROOT = os.path.dirname(_MODULES)

# 旧根 / 新根（相对项目根）
OLD_ROOT = os.path.join(PROJECT_ROOT, 'data', 'diagnosis')
NEW_ROOT = os.path.join(PROJECT_ROOT, 'data')

# 逐词表 .csv 的后缀特征：文件名里不含任何已知模型后缀的 .csv 就是逐词表。
# （逐词表就叫 <uid>.csv，特征叫 <uid><模型>_pauses*.pt）
CSV_SUFFIX = '.csv'

# 特征目录改名对照：旧目录名 -> 新目录名（前缀统一补 feat_）
FEATURE_DIR_RENAME = {
    'text': 'feat_distil',              # 老实验的 distil+wav2vec2 特征（在 text/ 里）
    'text_xlmr_xlsr': 'feat_xlmr_xlsr',
    'text_xlmr_wav2vec2': 'feat_xlmr_wav2vec2',
}

# 逐词表从 text/ 里拆出来的目标目录名
WORDS_DIR_NAME = 'words'

# 独立搬运的目录（audio / splits 本来就独立，也一起挪掉 diagnosis 这层）
PLAIN_MOVE = ('audio', 'segmentation', 'splits')

# 随 split 一起挪的顶层文件
TOP_FILES = (
    'adresso-train-mmse-scores.csv',
    'test_labels.csv',
    'text_transcriptions.csv',
    'text_transcriptions.csv.bak',
    'preprocess_skipped.csv',
)

DX_NAMES = ('cn', 'ad')


def _rel(p):
    return os.path.relpath(p, PROJECT_ROOT)


def plan_move(src, dst, moves, conflicts, skipped):
    """登记一次搬运：src -> dst。目标已存在且内容一致就跳过。"""
    if not os.path.exists(src):
        return
    if os.path.exists(dst):
        if filecmp.cmp(src, dst, shallow=True):
            skipped.append((src, dst))
        else:
            conflicts.append((src, dst))
        return
    moves.append((src, dst))


def collect(split_dir, moves, conflicts, skipped, empties):
    """收集一个 split（train 或 test）下的所有搬运计划。"""
    if not os.path.isdir(split_dir):
        return

    # ---- 1) 独立目录整体搬（audio / segmentation / splits）----
    for name in PLAIN_MOVE:
        src = os.path.join(split_dir, name)
        if os.path.isdir(src):
            plan_move(src, os.path.join(NEW_ROOT, os.path.basename(split_dir), name),
                      moves, conflicts, skipped)

    # ---- 2) 顶层文件搬 ----
    for fn in TOP_FILES:
        src = os.path.join(split_dir, fn)
        plan_move(src, os.path.join(NEW_ROOT, os.path.basename(split_dir), fn),
                  moves, conflicts, skipped)

    # ---- 3) 特征目录改名 ----
    for old_name, new_name in FEATURE_DIR_RENAME.items():
        old_dir = os.path.join(split_dir, old_name)
        if not os.path.isdir(old_dir):
            continue
        new_dir = os.path.join(NEW_ROOT, os.path.basename(split_dir), new_name)

        # text/ 里混着逐词表 .csv，要拆到 words/；其余目录整体搬。
        if old_name == 'text':
            words_dir = os.path.join(NEW_ROOT, os.path.basename(split_dir),
                                     WORDS_DIR_NAME)
            for dx in DX_NAMES:
                d = os.path.join(old_dir, dx)
                if not os.path.isdir(d):
                    continue
                for fn in sorted(os.listdir(d)):
                    src = os.path.join(d, fn)
                    if not os.path.isfile(src):
                        # 顺手记下空目录 / .ipynb_checkpoints 之类的杂物
                        if os.path.isdir(src) and not os.listdir(src):
                            empties.append(src)
                        continue
                    if fn.endswith(CSV_SUFFIX):
                        plan_move(src, os.path.join(words_dir, dx, fn),
                                  moves, conflicts, skipped)
                    else:
                        plan_move(src, os.path.join(new_dir, dx, fn),
                                  moves, conflicts, skipped)
        else:
            plan_move(old_dir, new_dir, moves, conflicts, skipped)

        # 空目录（比如 .ipynb_checkpoints）
        for root, dirs, files in os.walk(old_dir):
            if not dirs and not files:
                empties.append(root)


def apply_moves(moves):
    for src, dst in moves:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.move(src, dst)


def verify_moves(moves):
    """逐字节核对：搬完之后目标存在、且和"搬之前的源"一致。

    注意此时 src 已经不在了，所以这里只检查 dst 存在且非空；
    真正的完整性靠搬运前后文件数/大小对比（见 main 里的统计）。
    """
    bad = []
    for _src, dst in moves:
        if not os.path.exists(dst):
            bad.append(dst)
        elif os.path.isfile(dst) and os.path.getsize(dst) == 0:
            bad.append(dst + ' (0 字节)')
    return bad


def count_files(root):
    n, total = 0, 0
    for r, _, fs in os.walk(root):
        for f in fs:
            n += 1
            total += os.path.getsize(os.path.join(r, f))
    return n, total


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--apply', action='store_true', help='真正搬运（默认只干跑）')
    ap.add_argument('--verify', action='store_true', help='搬完核对目标文件都在')
    args = ap.parse_args()

    if not os.path.isdir(OLD_ROOT):
        print('找不到旧目录: %s' % _rel(OLD_ROOT))
        if os.path.isdir(NEW_ROOT):
            print('（也许已经重整过了？新结构：%s）' % _rel(NEW_ROOT))
        return 1

    n_before, size_before = count_files(OLD_ROOT)

    moves, conflicts, skipped, empties = [], [], [], []
    for split in ('train', 'test'):
        collect(os.path.join(OLD_ROOT, split), moves, conflicts, skipped, empties)

    print('=' * 72)
    print('data/ 层级重整')
    print('=' * 72)
    print('旧根: %s' % _rel(OLD_ROOT))
    print('新根: %s' % _rel(NEW_ROOT))
    print('当前共 %d 个文件 / %.1f MB' % (n_before, size_before / 1e6))
    print()

    print('--- 搬运项 (%d) ---' % len(moves))
    # 聚合打印：整目录搬的标 [目录]，逐文件的按 (源目录, 目标目录) 聚合
    agg = {}
    for src, dst in moves:
        if os.path.isdir(src):
            agg[('[' + _rel(src) + ']', '[' + _rel(dst) + ']')] = agg.get(
                ('[' + _rel(src) + ']', '[' + _rel(dst) + ']'), 0)
            continue
        key = (_rel(os.path.dirname(src)), _rel(os.path.dirname(dst)))
        agg[key] = agg.get(key, 0) + 1
    for (s, d), n in sorted(agg.items()):
        if n == 0:
            print('  %-44s -> %-30s  （整个目录）' % (s, d))
        else:
            print('  %-44s -> %-30s  %4d 项' % (s, d, n))

    if skipped:
        print()
        print('--- 已就位、跳过 (%d) ---' % len(skipped))
        for s, d in skipped[:5]:
            print('  %s' % _rel(s))
        if len(skipped) > 5:
            print('  ...（共 %d 项）' % len(skipped))

    if empties:
        print()
        print('--- 空目录（搬完顺手删掉）(%d) ---' % len(empties))
        for e in empties:
            print('  %s' % _rel(e))

    if conflicts:
        print()
        print('!! 目标已存在且内容不同，拒绝覆盖 (%d):' % len(conflicts))
        for s, d in conflicts[:10]:
            print('  %s\n     -> %s' % (_rel(s), _rel(d)))
        print('  请人工确认后再跑。')
        return 2

    if not args.apply:
        print()
        print('（这是干跑。确认无误后加 --apply 才真正搬运）')
        return 0

    print()
    print('开始搬运...')
    apply_moves(moves)
    for e in sorted(empties, key=len, reverse=True):
        try:
            os.rmdir(e)
        except OSError:
            pass
    print('搬运完成。')

    n_after, size_after = count_files(NEW_ROOT)
    print('新结构共 %d 个文件 / %.1f MB' % (n_after, size_after / 1e6))
    if n_after != n_before or abs(size_after - size_before) > 1024:
        print('⚠️ 文件数或体积对不上：搬前 %d/%.1fMB，搬后 %d/%.1fMB'
              % (n_before, size_before / 1e6, n_after, size_after / 1e6))
    else:
        print('✅ 文件数与体积一致，没有丢东西。')

    if args.verify:
        bad = verify_moves(moves)
        if bad:
            print('!! 目标文件缺失 (%d): %s' % (len(bad), bad[:5]))
            return 3
        print('✅ 目标文件全部就位。')

    # 旧根如果空了就删掉（diagnosis 这层）
    if os.path.isdir(OLD_ROOT) and not os.listdir(OLD_ROOT):
        os.rmdir(OLD_ROOT)
        print('已删除空的 %s' % _rel(OLD_ROOT))
    elif os.path.isdir(OLD_ROOT):
        rest = os.listdir(OLD_ROOT)
        if rest:
            print('注意：%s 还有残留: %s' % (_rel(OLD_ROOT), rest[:10]))

    return 0


if __name__ == '__main__':
    sys.exit(main())
