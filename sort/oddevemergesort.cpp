// https://stackoverflow.com/questions/34426337/how-to-fix-this-non-recursive-odd-even-merge-sort-algorithm
// Odd even merge sort - The Art of Computer Programming, vol 3 (algorithm 5.2.2M)

#include<iostream>
#include<stdlib.h>

#define log2length 6
#define length 64

void oddeven_merge_sort(int *unsorted){
    int p = 1 << (log2length - 1);
    int temp;

    for(int p = 1 << (log2length - 1); p > 0; p /= 2) {
        
        int q = 1 << log2length;
        int r = 0;

        for (int d = p ; d > 0 ; d = q - p) {
            for (int i = 0 ; i < length - d ; i++) {
                if ((i & p) == r) {
                    if (unsorted[i] > unsorted[i + d]){
                        temp = unsorted[i];
                        unsorted[i] = unsorted[i + d];
                        unsorted[i + d] = temp;
                    }
                }
            }
            q /= 2;
            r = p;
        }

    }
}

int main(){

    srand(0);

    int unsorted[length];
    for(int i = 0; i < length; i++){
        unsorted[i] = rand() & 15;
    }

    for(int i = 0; i < length; i++){
        std::cout<<unsorted[i]<< " ";
    }
    
    std::cout<<"\n";

    oddeven_merge_sort(unsorted);

    for(int i = 0; i < length; i++){
        std::cout<<unsorted[i]<< " ";
    }
    
    std::cout<<"\n";

    return 0;

}