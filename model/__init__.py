# -*- coding: utf-8 -*-
# @Time    : 2025/7/22 下午1:13
# @Author  : liguochun
# @FileName: __init__.py
# @Email   ：liguochun0304@163.com

import os
from typing import Optional


def _resolve_path(script_dir: str, path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    if os.path.isabs(path):
        return path if os.path.exists(path) else None
    for base in ("/root/autodl-fs", script_dir):
        cand = os.path.join(base, path)
        if os.path.exists(cand):
            return cand
    return None
