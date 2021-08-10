a = [[0 for _ in range(0,50)] for _ in range(0,50)]

for i in range(0, 512 + 1):
    a[i//16][i%16] = i


for i in range(0, 32):
    for j in range(0, 32):
        print(a[i][j], end=" ")
    print("\n")