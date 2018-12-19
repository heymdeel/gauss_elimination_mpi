import sys

import numpy as np
from mpi4py import MPI

# returns number of rows for proc with given rank
def get_chunk(total, world_size, rank):
    q = total // world_size
    
    if (total % world_size > 0):
        q += 1
    r = world_size * q - total

    chunk = q
    if (rank >= world_size - r):
        chunk = q - 1

    return chunk

inp = open('input.txt', 'r')

comm = MPI.COMM_WORLD
master = 0

n = int(inp.readline())
global_t = MPI.Wtime()

# rank and cluster size
rank = comm.rank
world_size = comm.size

nrows = get_chunk(n, world_size, rank)
rows = np.zeros(nrows, dtype = int) # indexes of rows for current proc

a, tmp, A, x = np.empty((nrows, n + 1)), np.empty(n + 1), [], np.empty(n)

t = MPI.Wtime()
if (rank == master):
    print("Reading data from file...")

a_ind, row = 0, 0
for i, line in enumerate(inp):
    split = [int(k) for k in line.split()]
    
    if (i % world_size == rank):
        a[a_ind] = split
        rows[row] = i
        row += 1
        a_ind += 1

    if (rank == master):
        A.append(split)

print("{}: {} => {}".format(rank, nrows, rows))

if (rank == master):
    t = MPI.Wtime() - t
    print("Reading data completed: time (sec) {0:.5}".format(t))
    t = MPI.Wtime()
    print("Forward elimination...")

# forward elimination
row = 0
for i in range(n-1):
    if (i in rows):
        tmp = np.copy(a[row])
        comm.Bcast(tmp, root = rank)
    
        row += 1
    else:
        comm.Bcast(tmp, root = i % world_size)
        
    for j in range(row, nrows):
        scaling = a[j][i] / tmp[i]
        for k in range(i, n + 1):
            a[j][k] -= scaling * tmp[k]

if (rank == master):
    t = MPI.Wtime() - t
    print("Forward elimination completed: time (sec) {0:.5}".format(t))
    t = MPI.Wtime()
    print("Back substitution...")

# back substitution
row = nrows - 1
for i in range(n-1, 0, -1):
    if (i in rows):
        tmp = np.copy(a[row])
        comm.Bcast(tmp, root = rank)
    
        row -= 1
    else:
        comm.Bcast(tmp, root = i % world_size)

    for j in range(row, -1, -1):
        scaling = a[j][i] / tmp[i]
        for k in range(i, n + 1):
            a[j][k] -= scaling * tmp[k]

if (rank == master):
    t = MPI.Wtime() - t
    print("Back substitution completed: time (sec) {0:.5}".format(t))
    print("Gathering result...")

# gathering results
for i in range(nrows):
    ind = rows[i]
    data = a[i][n] / a[i][ind]

    if (rank == master):
        x[ind] = data
    else:
        comm.send(data, dest = master, tag = ind)

if rank == master:
    for i in range(n):
        if (i not in rows):
            data = comm.recv(tag = i)
            x[i] = data

    global_t = MPI.Wtime() - global_t    
    print("Gaussian Elimination (MPI): n {}, procs {}, time (sec) {}\n".format(n, world_size, global_t))
    print("X ==", x)

    sum = 0
    for i in range(n):
        sum += x[i] * A[0][i]

    print("{} == {}".format(sum, A[0][n]))