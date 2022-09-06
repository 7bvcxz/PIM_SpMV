import torch

R = 10
C = 10

D = torch.zeros((R, C))

"""
for r_i in range(R):
    for c_i in range(C):
        if r_i < 5 and c_i < 5:
            D[r_i][c_i] = 1
        elif r_i >= 5 and c_i >= 5:
            D[r_i][c_i] = 1
"""
D = torch.Tensor([[1,1,1,1,0,0,0,0,0,0],
                  [1,1,1,1,0,0,0,0,0,0],
                  [0,0,1,1,1,1,0,0,0,0],
                  [0,0,1,1,1,1,0,0,0,0],
                  [0,0,0,0,1,1,1,1,0,0],
                  [0,0,0,0,1,1,1,1,0,0],
                  [0,0,0,0,1,1,1,1,0,0],
                  [0,0,0,0,0,0,1,1,1,1],
                  [0,0,0,0,0,0,1,1,1,1],
                  [0,0,0,0,0,0,1,1,1,1]])

for r_i in range(R):
    for c_i in range(C):
        if D[r_i][c_i] == 1:
            print("O", end=" ")
        else:
            print("X", end=" ")
    print()

torch.save(D, 'model.pt')


