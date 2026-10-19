#include <bits/stdc++.h>
#include <omp.h>
using namespace std;

#define int long long

struct Result {
    vector<double> w;
    double b = 0, seconds = 0;
    int threads = 1;
};

double soft(double z, double alpha) {
    if (z > alpha) return z - alpha;
    if (z < -alpha) return z + alpha;
    return 0;
}

void accumulate(vector<double>& x, vector<double>& y, vector<double>& w, double bias,
                vector<double>& gradient, int begin, int end, int d, bool simd) {
    double bias_gradient = 0;
    for (int i = begin; i < end; i++) {
        double prediction = 0;
        if (simd) {
            #pragma omp simd reduction(+:prediction)
            for (int j = 0; j < d; j++) prediction += x[i * d + j] * w[j];
        } else {
            for (int j = 0; j < d; j++) prediction += x[i * d + j] * w[j];
        }
        double error = prediction + bias - y[i];
        bias_gradient += error;
        if (simd) {
            #pragma omp simd
            for (int j = 0; j < d; j++) gradient[j] += error * x[i * d + j];
        } else {
            for (int j = 0; j < d; j++) gradient[j] += error * x[i * d + j];
        }
    }
    gradient[d] = bias_gradient;
}

Result train(vector<double>& x, vector<double>& y, int n, int d, int iterations,
             int threads, double lambda, double eta, string version) {
    bool parallel = version == "omp" || version == "omp_simd";
    bool simd = version == "simd" || version == "omp_simd";
    Result result;
    result.w.resize(d, 0);
    int workers = parallel ? threads : 1;
    vector<vector<double>> partial(workers, vector<double>(d + 1));
    double start = omp_get_wtime();
    for (int step = 0; step < iterations; step++) {
        for (int t = 0; t < workers; t++)
            fill(partial[t].begin(), partial[t].end(), 0);
        if (parallel) {
            #pragma omp parallel num_threads(threads)
            {
                int id = omp_get_thread_num();
                int count = omp_get_num_threads();
                if (id == 0) result.threads = count;
                accumulate(x, y, result.w, result.b, partial[id],
                           n * id / count, n * (id + 1) / count, d, simd);
            }
        } else {
            accumulate(x, y, result.w, result.b, partial[0], 0, n, d, simd);
        }
        for (int j = 0; j < d; j++) {
            double gradient = 0;
            for (int t = 0; t < result.threads; t++) gradient += partial[t][j];
            result.w[j] = soft(result.w[j] - eta * gradient / n, eta * lambda);
        }
        double gradient = 0;
        for (int t = 0; t < result.threads; t++) gradient += partial[t][d];
        result.b -= eta * gradient / n;
    }
    result.seconds = omp_get_wtime() - start;
    return result;
}

double loss(vector<double>& x, vector<double>& y, Result& result, int n, int d, double lambda) {
    double value = 0;
    for (int i = 0; i < n; i++) {
        double prediction = result.b;
        for (int j = 0; j < d; j++) prediction += x[i * d + j] * result.w[j];
        value += (prediction - y[i]) * (prediction - y[i]);
    }
    value /= 2 * n;
    for (double w : result.w) value += lambda * abs(w);
    return value;
}

void print(vector<double>& x, vector<double>& y, Result& result,
           int n, int d, int iterations, double lambda, double eta, string version,
           vector<double>& times, double max_dw, double max_db, double max_df, int mismatches) {
    vector<double> sorted = {times[0], times[1], times[2]};
    sort(sorted.begin(), sorted.end());
    int zeros = 0;
    for (double w : result.w) if (abs(w) < 1e-6) zeros++;
    double initial_loss = 0;
    for (double value : y) initial_loss += value * value / (2 * n);
    cout << "n,d,K,threads,lambda,eta,version,seconds,loss,initial_loss,zeros,dw,db,df,mismatches,t1,t2,t3\n";
    cout << setprecision(17) << n << ',' << d << ',' << iterations << ',' << result.threads
         << ',' << lambda << ',' << eta << ',' << version << ',' << sorted[1] << ','
         << loss(x, y, result, n, d, lambda) << ',' << initial_loss << ',' << zeros << ','
         << max_dw << ',' << max_db << ',' << max_df << ',' << mismatches;
    for (double t : times) cout << ',' << t;
    cout << '\n';
}

signed main(signed, char** argv) {
    int n = stoll(argv[1]), d = stoll(argv[2]), iterations = stoll(argv[3]);
    int threads = stoll(argv[4]);
    double lambda = stod(argv[5]), eta = stod(argv[6]);
    string version = argv[7];
    vector<double> x(n * d), y(n);
    mt19937 generator(42);
    uniform_real_distribution<double> feature(-sqrt(3.0), sqrt(3.0));
    uniform_real_distribution<double> noise(-0.05, 0.05);
    double truth[] = {1.5, -2.0, 0.5, 1.0};
    for (int i = 0; i < n; i++) {
        y[i] = 0.7 + noise(generator);
        for (int j = 0; j < d; j++) {
            x[i * d + j] = feature(generator);
            if (j < 4) y[i] += x[i * d + j] * truth[j];
        }
    }
    omp_set_dynamic(0);
    Result reference = train(x, y, n, d, iterations, 1, lambda, eta, "serial");
    double reference_loss = loss(x, y, reference, n, d, lambda);
    vector<double> times(3);
    double max_dw = 0, max_db = 0, max_df = 0;
    int mismatches = 0;
    Result result;
    for (int run = 0; run < 4; run++) {
        result = train(x, y, n, d, iterations, threads, lambda, eta, version);
        if (run > 0) times[run - 1] = result.seconds;
        double dw = 0;
        for (int j = 0; j < d; j++) dw = max(dw, abs(result.w[j] - reference.w[j]));
        double db = abs(result.b - reference.b);
        double value = loss(x, y, result, n, d, lambda);
        double df = abs(value - reference_loss);
        max_dw = max(max_dw, dw);
        max_db = max(max_db, db);
        max_df = max(max_df, df);
        if (!isfinite(value) || dw > 1e-8 || db > 1e-8 || df > 1e-8) mismatches++;
    }
    print(x, y, result, n, d, iterations, lambda, eta, version,
          times, max_dw, max_db, max_df, mismatches);
    return 0;
}
