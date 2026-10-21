#include <bits/stdc++.h>
#include <omp.h>
using namespace std;

#define int long long

int actual_threads = 1;

void ijk_seq(double* a, double* b, double* c, int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            double sum = 0;
            for (int k = 0; k < n; k++)
                sum += a[i * n + k] * b[k * n + j];
            c[i * n + j] = sum;
        }
}

void ijk_omp(double* a, double* b, double* c, int n) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        if (i == 0) actual_threads = omp_get_num_threads();
        for (int j = 0; j < n; j++) {
            double sum = 0;
            for (int k = 0; k < n; k++)
                sum += a[i * n + k] * b[k * n + j];
            c[i * n + j] = sum;
        }
    }
}

void ikj_seq(double* a, double* b, double* c, int n) {
    for (int i = 0; i < n; i++)
        for (int k = 0; k < n; k++) {
            double value = a[i * n + k];
            for (int j = 0; j < n; j++)
                c[i * n + j] += value * b[k * n + j];
        }
}

void ikj_omp(double* a, double* b, double* c, int n) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        if (i == 0) actual_threads = omp_get_num_threads();
        for (int k = 0; k < n; k++) {
            double value = a[i * n + k];
            for (int j = 0; j < n; j++)
                c[i * n + j] += value * b[k * n + j];
        }
    }
}

void blocked_omp(double* a, double* b, double* c, int n, int block) {
    #pragma omp parallel for collapse(2) schedule(static)
    for (int ii = 0; ii < n; ii += block)
        for (int jj = 0; jj < n; jj += block) {
            if (ii == 0 && jj == 0) actual_threads = omp_get_num_threads();
            for (int kk = 0; kk < n; kk += block)
                for (int i = ii; i < min(ii + block, n); i++)
                    for (int k = kk; k < min(kk + block, n); k++) {
                        double value = a[i * n + k];
                        for (int j = jj; j < min(jj + block, n); j++)
                            c[i * n + j] += value * b[k * n + j];
                    }
        }
}

signed main(signed, char** argv) {
    int n = stoll(argv[1]);
    int block = stoll(argv[2]);
    int threads = stoll(argv[3]);
    string version = argv[4];
    vector<double> a(n * n), b(n * n), c(n * n), reference(n * n);
    for (int k = 0; k < n * n; k++) {
        a[k] = ((k * 17 + 3) % 101 - 50) / 100.0;
        b[k] = ((k * 29 + 7) % 103 - 51) / 100.0;
    }
    ijk_seq(a.data(), b.data(), reference.data(), n);
    omp_set_dynamic(0);
    omp_set_num_threads(threads);
    double times[3], max_error = 0;
    int mismatches = 0;
    for (int run = -1; run < 3; run++) {
        fill(c.begin(), c.end(), 0);
        double start = omp_get_wtime();
        if (version == "ijk_seq") ijk_seq(a.data(), b.data(), c.data(), n);
        else if (version == "ijk_omp") ijk_omp(a.data(), b.data(), c.data(), n);
        else if (version == "ikj_seq") ikj_seq(a.data(), b.data(), c.data(), n);
        else if (version == "ikj_omp") ikj_omp(a.data(), b.data(), c.data(), n);
        else if (version == "blocked_omp") blocked_omp(a.data(), b.data(), c.data(), n, block);
        double seconds = omp_get_wtime() - start;
        if (run >= 0) times[run] = seconds;
        for (int k = 0; k < n * n; k++) {
            double error = abs(c[k] - reference[k]);
            max_error = max(max_error, error);
            if (!isfinite(c[k]) || error > 1e-9 + 1e-9 * abs(reference[k])) mismatches++;
        }
    }
    vector<double> sorted(times, times + 3);
    sort(sorted.begin(), sorted.end());
    cout << "n,block,threads,version,seconds,max_error,mismatches,t1,t2,t3\n";
    cout << setprecision(17) << n << ',' << block << ',' << actual_threads << ','
         << version << ',' << sorted[1] << ',' << max_error << ',' << mismatches;
    for (double t : times) cout << ',' << t;
    cout << '\n';
    return mismatches == 0 ? 0 : 1;
}
