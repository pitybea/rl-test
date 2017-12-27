import numpy as np

actions = [0, 1, 2] #rock scissors paper

def a1_wins(a1, a2):
    r = a1 - a2
    if r == -2:
        r = 1
    if r == 2:
        r = -1
    return -r

def sample(prob):
    p = np.random.rand()
    assert(np.abs(np.sum(prob) - 1.0) < 1e-7)
    acc = prob[0]
    i = 0
    while i < len(prob):
        if acc >= p:
            return i
        i += 1
        acc += prob[i]        
    return min(i, len(prob) - 1)
    
def regret_match(rs):
    s = np.sum(rs)
    if s > 0:
        return [r / s for r in rs]
    return [1.0 / len(rs)] * len(rs)

def regret_update(reg, i, v):
    reg[i] = max(0.0, p1_regret[i] + v)

if __name__ == '__main__':
    p1_regret = [0.0, 0.0, 0.0]
    p2_regret = [0.0, 0.0, 0.0]
    
    for _ in range(10):
        win_counts = 0
        lose_counts = 0 
        rounds = 100000
        for _ in range(rounds):
            a1 = actions[sample(regret_match(p1_regret))]
            a2 = actions[sample(regret_match(p2_regret))]
            std = a1_wins(a1, a2)
            win_counts += 1 if std > 0 else 0
            lose_counts += 1 if std < 0 else 0
            for reg, buf in zip([p1_regret, p2_regret], [(True, std), (False, -std)]):
                is_p1, cstd = buf
                for i in range(len(actions)):
                    a = actions[i]
                    v = (a1_wins(a, a2) if is_p1 else -a1_wins(a1, a)) - cstd
                    regret_update(reg, i, v)
        print 1.0 * win_counts / rounds, 1.0 * lose_counts / rounds
        print p1_regret, p2_regret
