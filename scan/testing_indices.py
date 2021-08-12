"""a = [[0 for _ in range(0,50)] for _ in range(0,50)]

for i in range(0, 512 + 1):
    a[i//16][i%16] = i


for i in range(0, 32):
    for j in range(0, 32):
        print(a[i][j], end=" ")
    print("\n")"""


print("padding when writing\n")
padding = 0
elements = 4
columns = []
found_padding = 1
while(padding < 32):
    columns = []
    column = 0
    found_padding = 1
    for thread in range(0, 32):
        if column not in columns:
            columns.append(column)
            column = column + 1
            if (thread + 1)%elements == 0:
                column = column + padding + 1
            if column > 32:
                column = column%32
        else :
            found_padding = 0
            break
    if found_padding:
        break
    padding = padding + 1 

print(padding)
import numpy as np
print(columns)
print(np.sort(columns))

print("padding when doing the warp serial sum\n")
elements = 5
columns = []
for thread in range(0, 32):
    column = ((thread * (elements + padding)))%32
    columns.append(column)

import numpy as np
print(columns)
print(np.sort(columns))
