# -*- coding: utf-8 -*-
"""打印某个已训练模型的参数规模（从仓库根目录搬过来的那个入口脚本）。

底层 `core.utils.get_model_statistics()` 现在直接用目录名做实验标识
（池化默认 mean、不写进目录名），不再按 `_` 切成两段。

用法
----
    cd cognialign && python tools/model_statistics.py <结果目录名>
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # cognialign/

from core.utils import get_model_statistics  # noqa: E402

if __name__ == "__main__":
    try:
        model_name = sys.argv[1]
        get_model_statistics(model_name)
    except IndexError:
        print("No model provided.")
        get_model_statistics()
