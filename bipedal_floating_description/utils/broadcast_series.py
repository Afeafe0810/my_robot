import pandas as pd
import numpy as np

class BroadcastSeries(pd.Series):
    """
    此類別繼承自 pandas.Series，
    其每個元素預期皆為 numpy 陣列，
    並且覆寫運算符號使得：
      1. 若與 scalar、numpy 陣列等運算，會對每個元素（即每個 key 的值）做運算
      2. 若與另一個 Series 運算，則會要求 index 完全匹配，並針對同 key 進行運算
    """
    # 設定 _metadata 讓屬性能夠正確傳遞到新建的物件中
    _metadata = []
    
    @property
    def _constructor(self):
        # 保證回傳的物件型態仍為 SeriesVec
        return BroadcastSeries

    def _apply_op(self, other, op):
        """
        內部共用方法，處理「左側」運算，即 self op other。
        若 other 為 Series 則要求 index 完全一致，
        否則視為 scalar 或 numpy 陣列，對 self 每個元素應用該運算。
        """
        if isinstance(other, pd.Series):
            # 檢查 index 是否完全一致
            if not self.index.equals(other.index):
                raise KeyError("兩個 Series 的 index 不匹配。")
            new_data = [op(self.loc[k], other.loc[k]) for k in self.index]
        else:
            new_data = [op(self.loc[k], other) for k in self.index]
        return self._constructor(new_data, index=self.index)

    def _r_apply_op(self, other, op):
        """
        內部共用方法，處理「右側」運算，即 other op self。
        """
        if isinstance(other, pd.Series):
            if not self.index.equals(other.index):
                raise KeyError("兩個 Series 的 index 不匹配。")
            new_data = [op(other.loc[k], self.loc[k]) for k in self.index]
        else:
            new_data = [op(other, self.loc[k]) for k in self.index]
        return self._constructor(new_data, index=self.index)

    # 加法
    def __add__(self, other):
        return self._apply_op(other, lambda a, b: a + b)
    def __radd__(self, other):
        return self._r_apply_op(other, lambda a, b: a + b)

    # 減法
    def __sub__(self, other):
        return self._apply_op(other, lambda a, b: a - b)
    def __rsub__(self, other):
        return self._r_apply_op(other, lambda a, b: a - b)

    # 逐元素乘法（element-wise multiplication）
    def __mul__(self, other):
        return self._apply_op(other, lambda a, b: a * b)
    def __rmul__(self, other):
        return self._r_apply_op(other, lambda a, b: a * b)

    # 矩陣乘法（@ 運算符）
    def __matmul__(self, other):
        return self._apply_op(other, lambda a, b: a @ b)
    def __rmatmul__(self, other):
        return self._r_apply_op(other, lambda a, b: a @ b)
