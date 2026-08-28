from collections import defaultdict

class EloModel:
    """
    Elo rating system with surface-specific tracking and recency decay.
    Players who haven't played recently gradually revert to 1500 rating.
    """
    def __init__(self, initial_rating=1500):
        self.initial_rating = initial_rating
        self.ratings = {}
        self.surface_ratings = {}
        self.match_counts = defaultdict(int)

    def get_rating(self, player, surface=None):
        """
        Get player's Elo rating (with decay for inactivity).
        
        Args:
            player: Player name
            surface: Court surface (Clay, Hard, Grass) or None for overall
            
        Returns:
            Current Elo rating
        """
        base = self.initial_rating
        
        if surface:
            raw = self.surface_ratings.setdefault(surface, {}).get(player, base)
        else:
            raw = self.ratings.get(player, base)
        
        # Recency decay: players who haven't played much revert toward 1500
        n = self.match_counts[player]
        decay = 0.999 ** max(0, 50 - n)
        
        return base + (raw - base) * decay

    def expected_score(self, r1, r2):
        """
        Calculate expected score using standard Elo formula.
        P(player1 wins) = 1 / (1 + 10^((r2 - r1) / 400))
        """
        return 1 / (1 + 10 ** ((r2 - r1) / 400))

    def update(self, p1, p2, p1_win, dom_ratio=0.5, surface=None, k=32):
        """
        Update Elo ratings based on match result.
        
        Args:
            p1: Player 1 name
            p2: Player 2 name
            p1_win: 1 if player1 won, 0 if player2 won
            dom_ratio: Dominance ratio (0.5-1.0) based on set/game score
            surface: Court surface for surface-specific rating update
            k: K-factor (higher = more volatile ratings)
        """
        # Adjust K-factor based on match dominance
        margin_mult = max(0.5, min(1.5, (dom_ratio - 0.5) * 3 + 0.8))
        k_adj = k * margin_mult
        
        # Update overall ratings
        r1, r2 = self.get_rating(p1), self.get_rating(p2)
        exp1 = self.expected_score(r1, r2)
        self.ratings[p1] = r1 + k_adj * (p1_win - exp1)
        self.ratings[p2] = r2 + k_adj * ((1 - p1_win) - (1 - exp1))
        self.match_counts[p1] += 1
        self.match_counts[p2] += 1
        
        # Update surface-specific ratings
        if surface:
            if surface not in self.surface_ratings:
                self.surface_ratings[surface] = {}
            
            sr1 = self.get_rating(p1, surface)
            sr2 = self.get_rating(p2, surface)
            exp_s = self.expected_score(sr1, sr2)
            self.surface_ratings[surface][p1] = sr1 + k_adj * (p1_win - exp_s)
            self.surface_ratings[surface][p2] = sr2 + k_adj * ((1 - p1_win) - (1 - exp_s))