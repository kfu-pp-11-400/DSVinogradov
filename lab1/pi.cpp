#define _USE_MATH_DEFINES
#include <bits/stdc++.h>
#include <omp.h>
using namespace std;

#define int long long

struct Result {
    double pi, seconds;
    int threads;
};

Result sequential(int n) {
    double sum = 0;
    double start = omp_get_wtime();
    for (int i = 0; i < n; i++) {
        sum += (i % 2 == 0 ? 1.0 : -1.0) / (2.0 * i + 1.0);
    }
    double pi = 4 * sum;
    return {pi, omp_get_wtime() - start, 1};
}

Result parallel(int n, int threads) {
    double sum = 0;
    int actual_threads = 0;
    double start = omp_get_wtime();
    #pragma omp parallel for reduction(+:sum) num_threads(threads) schedule(static)
    for (int i = 0; i < n; i++) {
        if (i == 0) actual_threads = omp_get_num_threads();
        sum += (i % 2 == 0 ? 1.0 : -1.0) / (2.0 * i + 1.0);
    }
    double pi = 4 * sum;
    return {pi, omp_get_wtime() - start, actual_threads};
}

void print(string mode, int n, Result r) {
    cout << mode << ',' << n << ',' << r.threads << ','
         << setprecision(17) << r.pi << ',' << abs(r.pi - M_PI)
         << ',' << r.seconds << '\n';
}

signed main() {
    int n, threads;
    cin >> n >> threads;
    omp_set_dynamic(0);
    cout << "mode,n,threads,pi,absolute_error,seconds\n";
    Result seq = sequential(n);
    Result par = parallel(n, threads);
    print("seq", n, seq);
    print("par", n, par);
    if (seq.seconds > 0 && par.seconds > 0)
        cerr << "Speedup: " << seq.seconds / par.seconds << '\n';
}

