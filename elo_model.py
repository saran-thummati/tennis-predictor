from collections import defaultdict

class EloModel:
    def __init__(self, initial_rating=1500):
        self.initial_rating  = initial_rating
        self.ratings         = {}
        self.surface_ratings = {}
        self.match_counts    = defaultdict(int)

    def get_rating(self, player, surface=None):
        base = self.initial_rating
        if surface: raw = self.surface_ratings.setdefault(surface, {}).get(player, base)
        else: raw = self.ratings.get(player, base)
        n = self.match_counts[player]
        decay = 0.999 ** max(0, 50 - n)
        return base + (raw - base) * decay

    def expected_score(self, r1, r2):
        return 1 / (1 + 10 ** ((r2 - r1) / 400))

    def update(self, p1, p2, p1_win, dom_ratio, surface=None, k=32):
        margin_mult = max(0.5, min(1.5, (dom_ratio - 0.5) * 3 + 0.8))
        k_adj = k * margin_mult
        r1, r2 = self.get_rating(p1), self.get_rating(p2)
        exp1   = self.expected_score(r1, r2)
        self.ratings[p1] = r1 + k_adj * (p1_win - exp1)
        self.ratings[p2] = r2 + k_adj * ((1 - p1_win) - (1 - exp1))
        self.match_counts[p1] += 1
        self.match_counts[p2] += 1
        if surface:
            sr1 = self.get_rating(p1, surface)
            sr2 = self.get_rating(p2, surface)
            exp_s = self.expected_score(sr1, sr2)
            self.surface_ratings[surface][p1] = sr1 + k_adj * (p1_win - exp_s)
            self.surface_ratings[surface][p2] = sr2 + k_adj * ((1 - p1_win) - (1 - exp_s))
