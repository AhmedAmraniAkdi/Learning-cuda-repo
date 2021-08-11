a = [[0 for _ in range(0,50)] for _ in range(0,50)]

for i in range(0, 512 + 1):
    a[i//16][i%16] = i


for i in range(0, 32):
    for j in range(0, 32):
        print(a[i][j], end=" ")
    print("\n")


print("padding\n")
# padding

padding = 0
elements = 5
columns = []
found_padding = 1
while(padding < 32):
    columns = []
    found_padding = 1
    for thread in range(0, 32):
        column = ((thread * (elements + padding)))%32
        if column not in columns:
            columns.append(column)
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
    
