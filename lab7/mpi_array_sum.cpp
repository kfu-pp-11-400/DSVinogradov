#include <bits/stdc++.h>
#include <mpi.h>
using namespace std;

struct Result {
    long long gather = 0, reduce = 0, allreduce = 0;
    double seconds = 0, gather_time = 0, reduce_time = 0;
    int correct = 0;
};

void print(int n, int size, Result result) {
    cout << "n,p,expected,gather,reduce,allreduce,correct,seconds,gather_time,reduce_time\n";
    cout << setprecision(10) << n << ',' << size << ',' << 1LL * n * (n + 1) / 2
         << ',' << result.gather << ',' << result.reduce << ',' << result.allreduce
         << ',' << result.correct << ',' << result.seconds
         << ',' << result.gather_time << ',' << result.reduce_time << '\n';
}

void compare(long long local, vector<long long>& sums, int rank, int size, Result& result) {
    int repeats = 1000;
    double start = 0;
    for (int i = 0; i < repeats + 10; i++) {
        if (i == 10) {
            MPI_Barrier(MPI_COMM_WORLD);
            start = MPI_Wtime();
        }
        MPI_Gather(&local, 1, MPI_LONG_LONG_INT, sums.data(), 1, MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            result.gather = 0;
            for (int r = 0; r < size; r++) result.gather += sums[r];
        }
    }
    double elapsed = (MPI_Wtime() - start) / repeats;
    MPI_Reduce(&elapsed, &result.gather_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    for (int i = 0; i < repeats + 10; i++) {
        if (i == 10) {
            MPI_Barrier(MPI_COMM_WORLD);
            start = MPI_Wtime();
        }
        MPI_Reduce(&local, &result.reduce, 1, MPI_LONG_LONG_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    }
    elapsed = (MPI_Wtime() - start) / repeats;
    MPI_Reduce(&elapsed, &result.reduce_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size, n = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    if (rank == 0 && argc > 1) n = atoi(argv[1]);
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (n <= 0 || n % size != 0) {
        if (rank == 0) cout << "N must be positive and divisible by the number of processes\n";
        MPI_Finalize();
        return 0;
    }
    vector<int> data;
    if (rank == 0) {
        data.resize(n);
        iota(data.begin(), data.end(), 1);
    }
    vector<int> part(n / size);
    vector<long long> sums(size);
    MPI_Scatter(data.data(), n / size, MPI_INT, part.data(), n / size, MPI_INT, 0, MPI_COMM_WORLD);
    long long local = 0;
    for (int value : part) local += value;
    Result result;
    MPI_Gather(&local, 1, MPI_LONG_LONG_INT, sums.data(), 1, MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);
    if (rank == 0)
        for (long long value : sums) result.gather += value;
    MPI_Reduce(&local, &result.reduce, 1, MPI_LONG_LONG_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Allreduce(&local, &result.allreduce, 1, MPI_LONG_LONG_INT, MPI_SUM, MPI_COMM_WORLD);
    double elapsed = MPI_Wtime() - start;
    MPI_Reduce(&elapsed, &result.seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (argc > 2 && string(argv[2]) == "compare") compare(local, sums, rank, size, result);
    long long expected = 1LL * n * (n + 1) / 2;
    int correct = result.allreduce == expected;
    if (rank == 0) correct = correct && result.gather == expected && result.reduce == expected;
    MPI_Allreduce(&correct, &result.correct, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if (rank == 0) print(n, size, result);
    MPI_Finalize();
    return 0;
}
