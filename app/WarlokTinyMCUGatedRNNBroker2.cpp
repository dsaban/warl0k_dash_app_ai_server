// Gated RNN in C++ with BPTT + Noise + Dropout
#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <ctime>

using namespace std;

const double LR = 0.01;
const double DROPOUT_RATE = 0.2;

// Activation functions
double tanh_(double x) { return tanh(x); }
double dtanh(double x) { double t = tanh_(x); return 1 - t * t; }

// Random helpers
double randn() {
    static random_device rd;
    static mt19937 gen(rd());
    static normal_distribution<> d(0.0, 1.0);
    return d(gen);
}
int argmax(const vector<double>& v) {
    return distance(v.begin(), max_element(v.begin(), v.end()));
}
vector<double> softmax(const vector<double>& x) {
    double maxVal = *max_element(x.begin(), x.end());
    vector<double> exps(x.size());
    double sum = 0;
    for (size_t i = 0; i < x.size(); ++i) {
        exps[i] = exp(x[i] - maxVal);
        sum += exps[i];
    }
    for (double& val : exps) val /= sum;
    return exps;
}

struct RNN {
    int input_dim, hidden_dim, output_dim;
    vector<vector<double>> Wxh, Whh, Why;
    vector<double> bh, by;

    RNN(int in_d, int h_d, int out_d) : input_dim(in_d), hidden_dim(h_d), output_dim(out_d) {
        Wxh.resize(hidden_dim, vector<double>(input_dim));
        Whh.resize(hidden_dim, vector<double>(hidden_dim));
        Why.resize(output_dim, vector<double>(hidden_dim));
        bh.resize(hidden_dim);
        by.resize(output_dim);
        for (auto& row : Wxh) for (double& v : row) v = randn() * 0.1;
        for (auto& row : Whh) for (double& v : row) v = randn() * 0.1;
        for (auto& row : Why) for (double& v : row) v = randn() * 0.1;
    }

    double forward_backward(const vector<vector<double>>& input_seq, const vector<int>& target_seq) {
        size_t T = input_seq.size();
        vector<vector<double>> hs(T + 1, vector<double>(hidden_dim));
        vector<vector<double>> ys(T);
        vector<vector<double>> probs(T);
        double total_loss = 0.0;

        // Forward
        for (size_t t = 0; t < T; ++t) {
            for (int i = 0; i < hidden_dim; ++i) {
                hs[t+1][i] = bh[i];
                for (int j = 0; j < input_dim; ++j) hs[t+1][i] += Wxh[i][j] * input_seq[t][j];
                for (int j = 0; j < hidden_dim; ++j) hs[t+1][i] += Whh[i][j] * hs[t][j];
                hs[t+1][i] = tanh_(hs[t+1][i]);
            }
            ys[t].resize(output_dim);
            for (int i = 0; i < output_dim; ++i) {
                ys[t][i] = by[i];
                for (int j = 0; j < hidden_dim; ++j) ys[t][i] += Why[i][j] * hs[t+1][j];
            }
            probs[t] = softmax(ys[t]);
            total_loss -= log(probs[t][target_seq[t]] + 1e-9);
        }

        // Backward
        vector<vector<double>> dWxh(hidden_dim, vector<double>(input_dim));
        vector<vector<double>> dWhh(hidden_dim, vector<double>(hidden_dim));
        vector<vector<double>> dWhy(output_dim, vector<double>(hidden_dim));
        vector<double> dbh(hidden_dim), dby(output_dim);
        vector<double> dh_next(hidden_dim);

        for (int t = T - 1; t >= 0; --t) {
            vector<double> dy = probs[t];
            dy[target_seq[t]] -= 1.0;
            for (int i = 0; i < output_dim; ++i) {
                dby[i] += dy[i];
                for (int j = 0; j < hidden_dim; ++j)
                    dWhy[i][j] += dy[i] * hs[t+1][j];
            }
            vector<double> dh(hidden_dim);
            for (int j = 0; j < hidden_dim; ++j) {
                for (int i = 0; i < output_dim; ++i)
                    dh[j] += Why[i][j] * dy[i];
                dh[j] += dh_next[j];
                double dt = dtanh(hs[t+1][j]) * dh[j];
                dbh[j] += dt;
                for (int i = 0; i < input_dim; ++i)
                    dWxh[j][i] += dt * input_seq[t][i];
                for (int i = 0; i < hidden_dim; ++i)
                    dWhh[j][i] += dt * hs[t][i];
                for (int i = 0; i < hidden_dim; ++i)
                    dh_next[i] = Whh[i][j] * dt;
            }
        }

        for (int i = 0; i < hidden_dim; ++i) {
            bh[i] -= LR * dbh[i];
            for (int j = 0; j < input_dim; ++j)
                Wxh[i][j] -= LR * dWxh[i][j];
            for (int j = 0; j < hidden_dim; ++j)
                Whh[i][j] -= LR * dWhh[i][j];
        }
        for (int i = 0; i < output_dim; ++i) {
            by[i] -= LR * dby[i];
            for (int j = 0; j < hidden_dim; ++j)
                Why[i][j] -= LR * dWhy[i][j];
        }

        return total_loss;
    }

    string predict(const vector<vector<double>>& input_seq, const string& vocab) {
        vector<double> h(hidden_dim);
        vector<bool> dropout_mask(hidden_dim);
        for (int i = 0; i < hidden_dim; ++i)
            dropout_mask[i] = ((rand() % 100) >= DROPOUT_RATE * 100);

        string result = "";
        for (auto& x : input_seq) {
            vector<double> h_new(hidden_dim);
            for (int i = 0; i < hidden_dim; ++i) {
                h_new[i] = bh[i];
                for (int j = 0; j < input_dim; ++j) h_new[i] += Wxh[i][j] * x[j];
                for (int j = 0; j < hidden_dim; ++j) h_new[i] += Whh[i][j] * h[j];
                h_new[i] = tanh_(h_new[i]);
                if (!dropout_mask[i]) h_new[i] = 0.0;
            }
            h = h_new;

            vector<double> logits(output_dim);
            for (int i = 0; i < output_dim; ++i) {
                logits[i] = by[i];
                for (int j = 0; j < hidden_dim; ++j)
                    logits[i] += Why[i][j] * h[j];
            }
            int idx = argmax(softmax(logits));
            result += vocab[idx];
        }
        return result;
    }
};

int main() {
    // num of epochs
    int epochs = 3000;
    // target size
    int target_size = 32;
    string vocab = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
    string clean_target = "";
//    add timing to process
    cout << "[Starting RNN Training with Noise and Dropout]" << endl;
//     timing
    auto start = chrono::high_resolution_clock::now();

    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(0, vocab.size() - 1);
    for (int i = 0; i < target_size; ++i) {
        clean_target += vocab[dis(gen)];
    }

    string target = clean_target;
    // print target
    cout << "[Clean Target]: " << clean_target << endl;

    int noise_idx = rand() % 5;
    char noisy_char = vocab[rand() % vocab.size()];
    while (noisy_char == target[noise_idx])
        noisy_char = vocab[rand() % vocab.size()];
    target[noise_idx] = noisy_char;

    int input_dim = vocab.size();
    vector<vector<double>> x_seq;
    vector<int> t_seq;
    for (char c : target) {
        vector<double> onehot(input_dim, 0);
        onehot[vocab.find(c)] = 1.0;
        x_seq.push_back(onehot);
        t_seq.push_back(vocab.find(c));
    }

    RNN net(input_dim, 32, input_dim);
    for (int epoch = 0; epoch <= epochs; ++epoch) {
        double loss = net.forward_backward(x_seq, t_seq);
        if (epoch % 100 == 0)
            cout << "[Epoch " << epoch << "] Loss: " << loss << " | Reconstructed: " << net.predict(x_seq, vocab) << endl;
    }
    cout << "[Clean Target ]: " << clean_target << endl;
    cout << "[Noisy Version]: " << target << endl;
//    end timing
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    cout << "[Training Time]: " << duration.count() << " ms" << endl;

    // compare target and prediction
    //    show prediction
    string prediction = net.predict(x_seq, vocab);
    cout << "[Final Prediction]: " << prediction << endl;
    if (clean_target == net.predict(x_seq, vocab))
        cout << "Target and prediction match!" << endl;
    else
        cout << "X - Target and prediction do not match!" << endl;


    size_t ram_bytes = input_dim * 32 * sizeof(double) + 32 * 32 * sizeof(double) + input_dim * 32 * sizeof(double) + 32 * sizeof(double) + input_dim * sizeof(double);
    cout << "[Estimated AI RAM Usage]: " << (ram_bytes / 1024.0) << " KB" << endl;
    return 0;
}
