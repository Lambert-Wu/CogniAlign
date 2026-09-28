# -*- coding: utf-8 -*-
"""打印某个已训练模型的参数规模（从仓库根目录搬过来的那个入口脚本）。

⚠️ 注意：底层 `core.utils.get_model_statistics()` 按 `folder_name.split('_')`
期望结果目录名是两段，而实际目录是四段（如 distil_wav2vec2_cross_mean），
所以对所有现役目录都会跳过、打印不出东西。要用得先修那个函数。

用法
----
    cd modules && python tools/model_statistics.py <结果目录名>
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # modules/

from core.utils import get_model_statistics  # noqa: E402

if __name__ == "__main__":
    try:
        model_name = sys.argv[1]
        get_model_statistics(model_name)
    except IndexError:
        print("No model provided.")
        get_model_statistics()
