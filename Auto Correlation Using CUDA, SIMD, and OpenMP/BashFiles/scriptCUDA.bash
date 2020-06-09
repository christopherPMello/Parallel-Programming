for block in 16 32 64 128
do
    /usr/local/apps/cuda/cuda-10.1/bin/nvcc -o out mainCUDA.cu -DBLOCKSIZE=$block
    ./out
done
