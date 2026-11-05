#include <bits/stdc++.h>
#include <mpi.h>
#include <omp.h>
using namespace std;

struct Graph {
    int n, m;
    vector<int> offsets, edges;
};

struct Result {
    vector<int> dist, fronts;
    long long processed = 0;
    double seconds = 0, communication = 0;
};

Graph read(string file) {
    ifstream input(file);
    Graph g;
    input >> g.n >> g.m;
    vector<pair<int, int>> pairs(g.m);
    g.offsets.resize(g.n + 1);
    g.edges.resize(g.m);
    for (auto& edge : pairs) {
        input >> edge.first >> edge.second;
        g.offsets[edge.first + 1]++;
    }
    for (int v = 1; v <= g.n; v++) g.offsets[v] += g.offsets[v - 1];
    vector<int> position = g.offsets;
    for (auto edge : pairs) g.edges[position[edge.first]++] = edge.second;
    return g;
}

Result sequential(Graph& g, int source) {
    Result result;
    result.dist.assign(g.n, -1);
    vector<int> queue(g.n);
    int head = 0, tail = 0;
    double start = MPI_Wtime();
    queue[tail++] = source;
    result.dist[source] = 0;
    while (head < tail) {
        int v = queue[head++];
        for (int j = g.offsets[v]; j < g.offsets[v + 1]; j++) {
            int u = g.edges[j];
            if (result.dist[u] == -1) {
                result.dist[u] = result.dist[v] + 1;
                queue[tail++] = u;
            }
        }
    }
    result.seconds = MPI_Wtime() - start;
    return result;
}

Result parallel(Graph& g, int source, int rank, int size) {
    Result result;
    result.dist.assign(g.n, -1);
    vector<int> frontier, next, local(g.n), global(g.n);
    vector<vector<int>> candidates(omp_get_max_threads());
    frontier.reserve(g.n);
    next.reserve(g.n);
    result.fronts.reserve(g.n);
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    frontier.push_back(source);
    result.dist[source] = 0;
    int level = 0;
    while (!frontier.empty()) {
        result.fronts.push_back(frontier.size());
        int count = ((int)frontier.size() + size - 1 - rank) / size;
        result.processed += count;
        for (auto& list : candidates) list.clear();
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            #pragma omp for schedule(runtime)
            for (int k = 0; k < count; k++) {
                int v = frontier[rank + k * size];
                for (int j = g.offsets[v]; j < g.offsets[v + 1]; j++) {
                    int u = g.edges[j];
                    if (result.dist[u] == -1) candidates[tid].push_back(u);
                }
            }
        }
        fill(local.begin(), local.end(), 0);
        for (auto& list : candidates)
            for (int u : list) local[u] = 1;
        double exchange = MPI_Wtime();
        MPI_Allreduce(local.data(), global.data(), g.n, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        result.communication += MPI_Wtime() - exchange;
        next.clear();
        for (int u = 0; u < g.n; u++) {
            if (global[u] && result.dist[u] == -1) {
                result.dist[u] = level + 1;
                next.push_back(u);
            }
        }
        frontier.swap(next);
        level++;
    }
    double elapsed = MPI_Wtime() - start;
    MPI_Reduce(&elapsed, &result.seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    elapsed = result.communication;
    MPI_Reduce(&elapsed, &result.communication, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    return result;
}

void print(Graph& g, int source, int size, Result& seq, Result& par, int errors, vector<long long>& work) {
    int unreachable = count(par.dist.begin(), par.dist.end(), -1);
    double speedup = seq.seconds / par.seconds;
    cout << setprecision(10);
    cout << "n,m,source,p,t,seq,par,communication,speedup,efficiency,levels,unreachable,errors\n";
    cout << g.n << ',' << g.m << ',' << source << ',' << size << ',' << omp_get_max_threads()
         << ',' << seq.seconds << ',' << par.seconds << ',' << par.communication
         << ',' << speedup << ',' << speedup / (size * omp_get_max_threads())
         << ',' << par.fronts.size() << ',' << unreachable << ',' << errors << '\n';
    cout << "fronts";
    for (int value : par.fronts) cout << ',' << value;
    cout << "\nwork";
    for (long long value : work) cout << ',' << value;
    cout << "\ndist";
    if (g.n <= 20) for (int value : par.dist) cout << ',' << value;
    cout << '\n';
}

int main(int argc, char** argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (provided < MPI_THREAD_FUNNELED) MPI_Abort(MPI_COMM_WORLD, 1);
    omp_set_dynamic(0);
    Graph g = read(argv[1]);
    int source = stoi(argv[2]);
    Result seq = sequential(g, source);
    Result par = parallel(g, source, rank, size);
    int errors = 0, max_errors;
    for (int v = 0; v < g.n; v++) errors += seq.dist[v] != par.dist[v];
    MPI_Reduce(&errors, &max_errors, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    vector<long long> work(size);
    MPI_Gather(&par.processed, 1, MPI_LONG_LONG_INT, work.data(), 1, MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);
    if (rank == 0) print(g, source, size, seq, par, max_errors, work);
    MPI_Finalize();
    return 0;
}
