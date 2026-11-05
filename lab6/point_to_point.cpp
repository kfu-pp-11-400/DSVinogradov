#include <bits/stdc++.h>
#include <mpi.h>
using namespace std;

const int TAG_MESSAGE = 1;
const int TAG_REPLY = 2;

void print(int rank, string text) {
    cout << "Process " << rank << ": " << text << endl;
}

void exchange(int rank, int size) {
    char buffer[128];
    if (rank == 0) {
        for (int r = 1; r < size; r++) {
            string message = "Hello, process " + to_string(r);
            MPI_Send(message.c_str(), message.size() + 1, MPI_CHAR, r, TAG_MESSAGE, MPI_COMM_WORLD);
            print(rank, "sent to " + to_string(r) + ": " + message);
        }
        for (int r = 1; r < size; r++) {
            MPI_Recv(buffer, 128, MPI_CHAR, r, TAG_REPLY, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            print(rank, "reply from " + to_string(r) + ": " + buffer);
        }
    } else {
        MPI_Recv(buffer, 128, MPI_CHAR, 0, TAG_MESSAGE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        print(rank, "received from 0: " + string(buffer));
        string reply = "Message received by " + to_string(rank);
        MPI_Send(reply.c_str(), reply.size() + 1, MPI_CHAR, 0, TAG_REPLY, MPI_COMM_WORLD);
    }
}

void print(int bytes, int repeats, double seconds) {
    cout << bytes << ',' << repeats << ',' << setprecision(10) << seconds << '\n';
}

void benchmark(int rank) {
    if (rank > 1) return;
    int repeats = 1000;
    if (rank == 0) cout << "bytes,repeats,seconds\n";
    for (int bytes : {1, 1024, 65536, 1048576}) {
        vector<char> buffer(bytes, 'a');
        double start = 0;
        for (int i = 0; i < repeats + 10; i++) {
            if (rank == 0) {
                if (i == 10) start = MPI_Wtime();
                MPI_Send(buffer.data(), bytes, MPI_CHAR, 1, TAG_MESSAGE, MPI_COMM_WORLD);
                MPI_Recv(buffer.data(), bytes, MPI_CHAR, 1, TAG_REPLY, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            } else {
                MPI_Recv(buffer.data(), bytes, MPI_CHAR, 0, TAG_MESSAGE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Send(buffer.data(), bytes, MPI_CHAR, 0, TAG_REPLY, MPI_COMM_WORLD);
            }
        }
        if (rank == 0) print(bytes, repeats, (MPI_Wtime() - start) / repeats);
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size < 2) {
        print(rank, "Run with at least 2 processes");
    } else if (argc > 1 && string(argv[1]) == "time") {
        benchmark(rank);
    } else {
        exchange(rank, size);
    }
    MPI_Finalize();
    return 0;
}
