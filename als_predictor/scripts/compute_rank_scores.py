import numpy as np
from itertools import combinations
from scipy.stats import spearmanr, kendalltau


def extract_pairs(n):
    return list(zip(*combinations(range(n), 2)))

def compute_rank_scores(gold, pred, sizes):
    g = np.array(gold)
    p = np.array(pred)
    spearman = []
    kendall = []
    pairwise = []
    offset = 0
    
    for size in sizes:
        g = gold[offset:offset+size]
        p = pred[offset:offset+size]

        # Pairwise accuracy
        if len(g) > 1:
            # Spearman correlation coefficient
            try:
                spr = spearmanr(g, p).statistic
            except:
                spr = spearmanr(g, p).correlation
            if not np.isnan(spr):
                spearman.append(spr)

            # Kendall's tau
            kt = kendalltau(g, p)[0]
            if not np.isnan(kt):
                kendall.append(kt)
            
            first, second = extract_pairs(len(g))
            first = np.asarray(first)
            second = np.asarray(second)
            pair_g = (g[first] > g[second]).astype(int)
            pair_p = (p[first] > p[second]).astype(int)
            pairwise.extend((pair_g == pair_p).astype(int).tolist())
        offset += size 

    spearman = np.mean(np.array(spearman))
    kendall = np.mean(np.array(kendall))
    pairwise = np.mean(np.array(pairwise))
    return spearman, kendall, pairwise

if __name__ == '__main__':
    print(compute_rank_scores([0, 1, 2, 2], [1, 1, 1, 1], [1, 1, 1, 1]))
