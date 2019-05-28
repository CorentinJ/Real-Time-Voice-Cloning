from timeit import default_timer as timer
import numpy as np
import torch

# Layer:        18.3ms
# Dense:        18.3ms
# Sparse 100%:   2.1ms
# Sparse  99%:   3.0ms
# Sparse  98%:   6.1ms
# Sparse  97%:  10.9ms
# Sparse  96%:  12.1ms
# Sparse  95%:  14.2ms
# Sparse  90%:  22.5ms

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [torch.nn.Linear(10000, 10000).cuda() for _ in range(10)]
        # self.layers = [torch.Tensor(10000, 10000) for _ in range(10)]
        # self.layers = [torch.sparse.FloatTensor(10000, 10000).cuda() for _ in range(10)]
        
        for i in range(len(self.layers)):
            self.layers[i].weight.data = torch.sparse.FloatTensor(
                self.layers[i].weight.shape).cuda() 
            pass
        #     mask = torch.rand(self.layers[i].shape) > 0.90
        #     a = torch.sum(mask)
        #     print(a)
        #     self.layers[i][mask] = torch.randn(a)
        #     self.layers[i] = self.layers[i].to_sparse().cuda()

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            # x = torch.mm(layer, x)
            # x = torch.sparse.mm(layer, x)
        return x

def main():
    # Define model and input data
    model = Model().cuda()
    x = torch.from_numpy(np.random.rand(1, 10000).astype(np.float32)).cuda()

    # The first pass is always slower, so run it once
    model.forward(x)
    
    # Measure elapsed time
    passes = 40
    total_time = 0
    for _ in range(passes):
        torch.cuda.synchronize()
        start = timer()
        model(x)
        torch.cuda.synchronize()
        delta = timer() - start
        
        print('Forward pass: %.2fms' % (delta * 1000))
        total_time += delta
    print('Average forward pass: %.2fms' % (1000 * total_time / passes))
    
if __name__ == '__main__':
    main()