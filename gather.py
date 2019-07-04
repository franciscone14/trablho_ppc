from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

data = (rank+1)**2
print(rank, data)
data = comm.gather(data, root=0)
if rank == 0:
	print(rank, data)
    for i in range(size):
        assert data[i] == (i+1)**2
    print(rank, data)
else:
	print(rank, data)
    assert data is None
    print(rank, data)
print(rank, data)