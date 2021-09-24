import random as rnd

rnd.seed(0)

B = [i for i in range(0, 256)]
A = [i for i in range(0, 256)]
OUT = [0] * 512
d_input = A + B 

swap = 32//32
tempA = 0
tempB = 0
threadsperpair = 64 // 16

x = [32] * 32
y = [32] * 32
 

def seq_merge(Out, offset, A, tempA, x1, x2, B, tempB, y1, y2):

    if(offset == 4 * 16):
        print("**********************")
        print(A[tempA:tempA+16])
        print(B[tempB:tempB+16])
        print(x1, x2, y1, y2)
        print("**********************")


    item_A = A[tempA + x1]
    item_B = B[tempB + y1]



    for i in range(0, 16):

        p = (y1 < y2) and ((x1 >= x2) or item_B <= item_A)

        if(p):
            mergeitem = item_B; 
            y1 = y1 + 1
            item_B = B[tempB + y1];  
        else:
            mergeitem= item_A
            x1 = x1 + 1
            item_A = A[tempA + x1]; 
        
        if(offset == 4 * 16):
            print(p, mergeitem, item_B, item_A, "hello", y1, y2, x1, x2, item_B ,item_A)
        Out[i + offset] = mergeitem


def merge_some_threads():
    for thread in range(8,12):
        i = 0
        idx = thread
        threadIdx_perpair = thread % threadsperpair
        if(threadIdx_perpair != threadsperpair - 1):
            diag = (threadIdx_perpair + 1) * 32 * 2 // threadsperpair
            if diag > 32:
                atop = 32
                btop = diag - 32
            else :
                atop = diag
                btop = 0
            abot = btop

            tempA = thread // threadsperpair * 32
            tempB = thread // threadsperpair * 32

            while(1):
                offset = (atop - abot)//2
                ai = atop - offset
                bi = btop + offset

                if (ai >= 32 or bi == 0 or A[ai + tempA] > B[bi - 1 + tempB]):
                    if(ai == 0 or bi >= 32 or A[ai - 1 + tempA] <= B[bi + tempB]):
                        x[thread] = ai
                        y[thread] = bi
                        break
                    else :
                        atop = ai - 1
                        btop = bi + 1 
                else :
                    abot = ai + 1

        if (threadIdx_perpair == 0):
            x1 = 0
            y1 = 0
        else:
            x1 = x[thread - 1]
            y1 = y[thread - 1]

        seq_merge(OUT, thread * 16, A, tempA, x1, x[thread], B, tempB, y1, y[thread])


"""
merge_some_threads()
print(x)
print(y)
print("*********************")
print(OUT[64:128])
print("*********************")
print(OUT)
"""

A_t = []
B_t = []

def interleaved_access():
    i = 0
    offset_A = 0
    offset_B = 0

    for j in range(0, 16):
        if(swap & j):
            print("B")
            B_t[offset_B:offset_B+32] = d_input[(j*32):(j*32+32)]
            offset_B += 32

        else :
            print("A")
            A_t[offset_A:offset_A+32] = d_input[(j*32):(j*32+32)]
            offset_A += 32

"""
print("swap 1")
swap = 1
interleaved_access()
print("swap 2")
swap = 2
interleaved_access()
print("swap 4")
swap = 4
interleaved_access()
"""

interleaved_access()
print(A_t)
print(B_t)

