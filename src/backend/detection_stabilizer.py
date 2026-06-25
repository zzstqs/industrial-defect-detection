"""
检测结果稳定器 - 解决缺陷检测结果闪烁问题

原理：YOLO检测在帧与帧之间会产生波动（同一缺陷有时检测到有时检测不到），
通过在时间窗口内维持检测结果来避免前端列表闪烁。

策略：
1. 新检测结果到达 → 立即显示
2. 检测结果消失 → 维持显示 STICKY_FRAMES 帧后才清除
3. 检测结果变化 → 维持旧结果 STICKY_FRAMES 帧，新结果持续 CONFIRM_FRAMES 帧后切换
"""

from collections import OrderedDict
from typing import Dict, List, Optional, Any
import hashlib


class DetectionStabilizer:
    """
    检测结果稳定器

    核心思想：检测到的东西不会凭空消失，除非连续多帧都检测不到。
    这是工业检测中常用的时间滤波策略。
    """

    def __init__(
        self,
        sticky_frames: int = 8,      # 结果消失后保持显示的帧数
        confirm_frames: int = 3,     # 新结果需持续出现多少帧才显示
        max_history: int = 30,       # 最大历史帧数
    ):
        self.sticky_frames = sticky_frames
        self.confirm_frames = confirm_frames
        self.max_history = max_history

        # 历史缓冲区：存储最近 N 帧的检测结果
        self._history: List[List[Dict]] = []

        # 当前待显示的稳定结果
        self._current: List[Dict] = []
        self._current_key: str = ""

        # 每个检测目标ID的跟踪计数
        # key: 目标哈希, value: {"count": 连续出现帧数, "last_seen_at": 帧号, "data": 最近数据}
        self._tracker: OrderedDict = OrderedDict()

        # 帧计数器
        self._frame_idx: int = 0

    def feed(self, detections: List[Dict]) -> List[Dict]:
        """
        喂入新一帧的检测结果，返回稳定后的结果

        Args:
            detections: 当前帧的检测结果列表
                       每个元素需包含: id, class, confidence, bbox
        Returns:
            稳定后的检测结果列表
        """
        self._frame_idx += 1

        # 1. 将新帧加入历史
        self._history.append(detections)
        if len(self._history) > self.max_history:
            self._history.pop(0)

        # 2. 更新每个目标的跟踪状态
        current_keys = set()
        for d in detections:
            key = self._make_key(d)
            current_keys.add(key)

            if key in self._tracker:
                # 已存在的目标 → 更新计数和数据
                entry = self._tracker[key]
                entry["count"] = entry.get("count", 0) + 1
                entry["last_seen_at"] = self._frame_idx
                entry["data"] = d  # 更新为最新数据
                self._tracker.move_to_end(key)
            else:
                # 新出现的目标 → 记录
                self._tracker[key] = {
                    "count": 1,
                    "last_seen_at": self._frame_idx,
                    "data": d,
                }

        # 3. 标记未出现的旧目标
        for key, entry in self._tracker.items():
            if key not in current_keys:
                # 目标在当前帧未出现
                frames_since_last = self._frame_idx - entry["last_seen_at"]
                if frames_since_last > self.sticky_frames:
                    # 超过 STICKY_FRAMES 帧未出现 → 标记为过期
                    entry["expired"] = True

        # 4. 清理过期目标
        expired_keys = [k for k, v in self._tracker.items() if v.get("expired")]
        for k in expired_keys:
            del self._tracker[k]

        # 5. 筛选出有效的稳定结果（分配稳定ID）
        stable_results = []
        for key, entry in self._tracker.items():
            if entry["count"] >= self.confirm_frames:
                # 复制检测数据，并用稳定的 key 哈希作为固定 ID
                # 这样即使 YOLO 每帧给不同序号，前端也始终看到同一个 ID
                item = dict(entry["data"])
                item["id"] = hashlib.md5(key.encode()).hexdigest()[:6]
                stable_results.append(item)

        # 6. 检查是否与当前结果有变化
        new_key = self._stable_key(stable_results)
        if new_key != self._current_key:
            self._current = stable_results
            self._current_key = new_key

        return self._current

    def _make_key(self, detection: Dict) -> str:
        """基于检测框生成唯一标识 — 80px粗粒度网格，容忍YOLO框±40px抖动"""
        bbox = detection.get("bbox", [0, 0, 0, 0])
        cls_name = detection.get("class", "")
        # bbox中心
        cx = (bbox[0] + bbox[2]) // 2
        cy = (bbox[1] + bbox[3]) // 2
        # 宽高区间（用于区分同类别不同大小的缺陷）
        bw = (bbox[2] - bbox[0]) // 80
        bh = (bbox[3] - bbox[1]) // 80
        # 量化到 80px 网格，允许 40-50px 的位置偏移（YOLO 典型抖动范围）
        gx = cx // 80
        gy = cy // 80
        return f"{cls_name}_{gx}_{gy}_{bw}_{bh}"

    def _stable_key(self, results: List[Dict]) -> str:
        """生成稳定结果的标识 — 基于ID集合，完全不依赖bbox坐标
        因为已经在步骤5中分配了稳定ID（基于_make_key），
        只需要比较ID集合是否相同即可判断结果是否变化"""
        ids = sorted([d.get("id", d.get("class", "?")) for d in results])
        return "|".join(ids)

    def reset(self):
        """重置稳定器状态"""
        self._history.clear()
        self._current = []
        self._current_key = ""
        self._tracker.clear()
        self._frame_idx = 0

    def get_stats(self) -> Dict[str, Any]:
        """获取稳定器统计信息"""
        return {
            "frame_idx": self._frame_idx,
            "tracked_objects": len(self._tracker),
            "stable_results": len(self._current),
            "history_frames": len(self._history),
        }
