# CUDA excercise

### Reduction

```C
__global__ void reduce_sum(const float* input, float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N) {
        atomicAdd(output, input[idx]);
    }
}
```

단순한 atomicAdd의 경우 결과값이 determistic하지 않은데 atomicAdd는 결합법칙이 성립하지않아 더하는 순서에 따라 결과가 달라질수 있기 때문이라고 한다. 이를 방지하기 위해 구현할때는 shared memory를 사용하고 atomicAdd는 스레드 1개에서만 수행한다(첫번째 쓰레드)
이를 통해 오차를 줄이고 atomic 충돌 횟수를 줄일수잇다.