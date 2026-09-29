# -*- coding: UTF-8 -*-
"""pytest 公共引导：把仓库根目录加入 sys.path，保证 agent_ppo 可导入。

平台模块（kaiwudrl/common_python/tools.*）不在本地环境——本目录只收录
平台无关的静态测试，import 失败的模块一律用 pytest.importorskip 跳过。
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
