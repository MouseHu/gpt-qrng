def solve(K, piles):
    xor = 0
    for pile in piles:
        xor ^= pile % (K+1)
        xor = xor % (K+1)
    print(xor)
    if xor == 0:
        return 'LeapFrog'
    else:
        return 'Preempt'


datas = [(1, [1]), (30, [197943, 249832])]

for data in datas:
    print(solve(*data))
