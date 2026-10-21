#include <bits/stdc++.h>
#include <omp.h>
using namespace std;

#define int long long

struct Result {
    double value, seconds;
    int n, threads;
};

double f(double x) {
    return x * sin(x * x);
}

Result sequential(double a, double b, int n) {
    double start = omp_get_wtime();
    double h = (b - a) / n, sum = 0;
    for (int i = 0; i < n; i++) {
        double x = a + (i + 0.5) * h;
        sum += f(x);
    }
    double value = h * sum;
    return {value, omp_get_wtime() - start, n, 1};
}

Result parallel(double a, double b, int n, int threads) {
    double start = omp_get_wtime();
    double h = (b - a) / n, sum = 0;
    int actual_threads = 0;
    #pragma omp parallel for reduction(+:sum) num_threads(threads) schedule(static)
    for (int i = 0; i < n; i++) {
        if (i == 0) actual_threads = omp_get_num_threads();
        double x = a + (i + 0.5) * h;
        sum += f(x);
    }
    double value = h * sum;
    return {value, omp_get_wtime() - start, n, actual_threads};
}

Result accuracy(double a, double b, int n, int threads, double eps, bool use_parallel) {
    double start = omp_get_wtime();
    Result current = use_parallel ? parallel(a, b, n, threads) : sequential(a, b, n);
    while (true) {
        double previous = current.value;
        n *= 2;
        current = use_parallel ? parallel(a, b, n, threads) : sequential(a, b, n);
        if (abs(current.value - previous) < eps) break;
    }
    current.seconds = omp_get_wtime() - start;
    return current;
}

void print(string mode, Result r) {
    cout << mode << ',' << r.n << ',' << r.threads << ','
         << setprecision(17) << r.value << ',' << r.seconds << '\n';
}

signed main() {
    double a, b, eps;
    int n, threads;
    cin >> a >> b >> n >> threads >> eps;
    omp_set_dynamic(0);
    Result seq, par;
    if (eps == 0) {
        seq = sequential(a, b, n);
        par = parallel(a, b, n, threads);
    } else {
        seq = accuracy(a, b, n, threads, eps, false);
        par = accuracy(a, b, n, threads, eps, true);
    }
    cout << "mode,n,threads,integral,seconds\n";
    print("seq", seq);
    print("par", par);
    if (seq.seconds > 0 && par.seconds > 0)
        cerr << "Speedup: " << seq.seconds / par.seconds << '\n';
}
