class RankZeroOnly:
    """A class to handle rank zero operations"""
    def __init__(self):
        self._rank: int = 0
    
    @property
    def rank(self) -> int:
        return self._rank
    
    @rank.setter
    def rank(self, rank: int):
        self._rank = rank

rank_zero_only = RankZeroOnly()