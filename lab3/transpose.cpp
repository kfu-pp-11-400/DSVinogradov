#include <bits/stdc++.h>
#include <omp.h>
using namespace std;

#define int long long

int actual_threads = 1;

void transpose_rows(float* a, float* b, int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            b[j * n + i] = a[i * n + j];
}

void transpose_cols(float* a, float* b, int n) {
    for (int j = 0; j < n; j++)
        for (int i = 0; i < n; i++)
            b[j * n + i] = a[i * n + j];
}

void transpose_blocked(float* a, float* b, int n, int block) {
    for (int ii = 0; ii < n; ii += block)
        for (int jj = 0; jj < n; jj += block)
            for (int i = ii; i < min(ii + block, n); i++)
                for (int j = jj; j < min(jj + block, n); j++)
                    b[j * n + i] = a[i * n + j];
}

void transpose_rows_omp(float* a, float* b, int n) {
    #pragma omp parallel for schedule(runtime)
    for (int i = 0; i < n; i++) {
        if (i == 0) actual_threads = omp_get_num_threads();
        for (int j = 0; j < n; j++)
            b[j * n + i] = a[i * n + j];
    }
}

void transpose_blocked_omp(float* a, float* b, int n, int block) {
    #pragma omp parallel for schedule(static)
    for (int ii = 0; ii < n; ii += block) {
        if (ii == 0) actual_threads = omp_get_num_threads();
        for (int jj = 0; jj < n; jj += block)
            for (int i = ii; i < min(ii + block, n); i++)
                for (int j = jj; j < min(jj + block, n); j++)
                    b[j * n + i] = a[i * n + j];
    }
}

void transpose_inplace(float* a, int n) {
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++)
            swap(a[i * n + j], a[j * n + i]);
}

void transpose_inplace_omp(float* a, int n) {
    #pragma omp parallel for schedule(static, 1)
    for (int i = 0; i < n; i++) {
        if (i == 0) actual_threads = omp_get_num_threads();
        for (int j = i + 1; j < n; j++)
            swap(a[i * n + j], a[j * n + i]);
    }
}

signed main(signed argc, char** argv) {
    int n = stoll(argv[1]);
    int block = stoll(argv[2]);
    int threads = stoll(argv[3]);
    string version = argv[4];
    int chunk = 0;
    if (argc > 5) chunk = stoll(argv[5]);
    int count = n * n;
    float* a = (float*)malloc(count * sizeof(float));
    float* b = (float*)malloc(count * sizeof(float));
    float* reference = (float*)malloc(count * sizeof(float));
    if (!a || !b || !reference) {
        cout << "Not enough memory\n";
        free(a);
        free(b);
        free(reference);
        return 1;
    }
    for (int k = 0; k < count; k++) a[k] = k % 16777213;
    transpose_rows(a, reference, n);
    omp_set_dynamic(0);
    omp_set_num_threads(threads);
    omp_set_schedule(omp_sched_static, chunk);
    bool inplace = version == "inplace" || version == "inplace_omp";
    double times[5];
    int mismatches = 0;
    for (int run = -1; run < 5; run++) {
        if (inplace) copy(a, a + count, b);
        else fill(b, b + count, -1);
        double start = omp_get_wtime();
        if (version == "rows") transpose_rows(a, b, n);
        else if (version == "cols") transpose_cols(a, b, n);
        else if (version == "blocked") transpose_blocked(a, b, n, block);
        else if (version == "rows_omp") transpose_rows_omp(a, b, n);
        else if (version == "blocked_omp") transpose_blocked_omp(a, b, n, block);
        else if (version == "inplace") transpose_inplace(b, n);
        else if (version == "inplace_omp") transpose_inplace_omp(b, n);
        double seconds = omp_get_wtime() - start;
        if (run >= 0) times[run] = seconds;
        for (int k = 0; k < count; k++)
            if (b[k] != reference[k]) mismatches++;
    }
    vector<double> sorted(times, times + 5);
    sort(sorted.begin(), sorted.end());
    double seconds = sorted[2];
    cout << "n,block,threads,version,chunk,seconds,GB_per_second,mismatches,t1,t2,t3,t4,t5\n";
    cout << setprecision(17) << n << ',' << block << ',' << actual_threads << ','
         << version << ',' << chunk << ',' << seconds << ',';
    if (seconds > 0) cout << 2.0 * count * sizeof(float) / seconds / 1e9;
    else cout << "NA";
    cout << ',' << mismatches;
    for (double t : times) cout << ',' << t;
    cout << '\n';
    free(a);
    free(b);
    free(reference);
    return mismatches == 0 ? 0 : 1;
}
