#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <algorithm>
using namespace std;

const int vocab_size = 62;
const int hidden_dim = 8;
const int secret_len = 16;
const int epochs = 500;
const double lr = 0.05;

string vocab = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";

// Random one-hot sequence for secret
typedef vector<double> Vec;
typedef vector<Vec> Mat;

Vec one_hot(char c) {
    Vec vec(vocab_size, 0.0);
    int idx = vocab.find(c);
    if (idx >= 0) vec[idx] = 1.0;
    return vec;
}

Mat secret_to_tensor(const string& secret) {
    Mat tensor;
    for (char c : secret) tensor.push_back(one_hot(c));
    return tensor;
}

Vec softmax(const Vec& x) {
    Vec out(x.size());
    double max_x = *max_element(x.begin(), x.end());
    double sum = 0.0;
    for (int i = 0; i < x.size(); ++i) sum += (out[i] = exp(x[i] - max_x));
    for (double& v : out) v /= sum;
    return out;
}

class TinySecretRegenerator {
public:
    Mat w1, w2;
    Vec b1, b2;

    TinySecretRegenerator() {
        srand(time(0));
        w1.resize(vocab_size, Vec(hidden_dim));
        w2.resize(hidden_dim, Vec(vocab_size));
        b1.resize(hidden_dim, 0.0);
        b2.resize(vocab_size, 0.0);

        for (auto& row : w1) for (double& v : row) v = 0.1 * ((rand() / (double)RAND_MAX) - 0.5);
        for (auto& row : w2) for (double& v : row) v = 0.1 * ((rand() / (double)RAND_MAX) - 0.5);
    }

    Vec forward(const Vec& x) {
        Vec z1(hidden_dim), a1(hidden_dim);
        for (int i = 0; i < hidden_dim; ++i) {
            for (int j = 0; j < vocab_size; ++j)
                z1[i] += x[j] * w1[j][i];
            z1[i] += b1[i];
            a1[i] = tanh(z1[i]);
        }
        Vec z2(vocab_size);
        for (int i = 0; i < vocab_size; ++i) {
            for (int j = 0; j < hidden_dim; ++j)
                z2[i] += a1[j] * w2[j][i];
            z2[i] += b2[i];
        }
        return softmax(z2);
    }

    void train(const Mat& x, const Mat& y) {
        for (int epoch = 0; epoch <= epochs; ++epoch) {
            double total_loss = 0.0;
            for (int t = 0; t < secret_len; ++t) {
                Vec z1(hidden_dim), a1(hidden_dim);
                for (int i = 0; i < hidden_dim; ++i) {
                    for (int j = 0; j < vocab_size; ++j)
                        z1[i] += x[t][j] * w1[j][i];
                    z1[i] += b1[i];
                    a1[i] = tanh(z1[i]);
                }
                Vec z2(vocab_size), a2(vocab_size);
                double sum = 0.0;
                for (int i = 0; i < vocab_size; ++i) {
                    for (int j = 0; j < hidden_dim; ++j)
                        z2[i] += a1[j] * w2[j][i];
                    z2[i] += b2[i];
                    a2[i] = exp(z2[i]);
                    sum += a2[i];
                }
                for (double& v : a2) v /= sum;
                for (int i = 0; i < vocab_size; ++i)
                    total_loss += pow(a2[i] - y[t][i], 2);

                Vec dz2(vocab_size);
                for (int i = 0; i < vocab_size; ++i)
                    dz2[i] = 2 * (a2[i] - y[t][i]) / vocab_size;

                Vec da1(hidden_dim);
                for (int j = 0; j < hidden_dim; ++j) {
                    for (int i = 0; i < vocab_size; ++i)
                        w2[j][i] -= lr * dz2[i] * a1[j];
                    da1[j] = 0.0;
                    for (int i = 0; i < vocab_size; ++i)
                        da1[j] += dz2[i] * w2[j][i];
                    da1[j] *= 1 - a1[j] * a1[j];
                }
                for (int i = 0; i < vocab_size; ++i) b2[i] -= lr * dz2[i];
                for (int j = 0; j < hidden_dim; ++j) b1[j] -= lr * da1[j];
                for (int i = 0; i < vocab_size; ++i)
                    for (int j = 0; j < hidden_dim; ++j)
                        w1[i][j] -= lr * x[t][i] * da1[j];
            }
            if (epoch % 100 == 0)
                cout << "Epoch " << epoch << ", Loss: " << total_loss / secret_len << endl;
        }
    }
};

string decode(const Mat& output) {
    string out;
    for (const Vec& o : output) {
        int idx = max_element(o.begin(), o.end()) - o.begin();
        out += vocab[idx];
    }
    return out;
}

int main() {
    srand(time(0));
    string secret = "";
    for (int i = 0; i < secret_len; ++i)
        secret += vocab[rand() % vocab.size()];

    cout << "Generated Secret: " << secret << endl;
    Mat x = secret_to_tensor(secret);
    Mat y = x;

    TinySecretRegenerator model;
    model.train(x, y);

    Mat out;
    for (int i = 0; i < x.size(); ++i)
        out.push_back(model.forward(x[i]));

    string recon = decode(out);
    cout << "Reconstructed: " << recon << endl;
    cout << "Match: " << (recon == secret ? "✅" : "❌") << endl;

    size_t ram = x.size() * x[0].size() + hidden_dim * vocab_size + vocab_size * hidden_dim;
    cout << "Estimated RAM (approx): " << ram * sizeof(double) / 1024.0 << " KB" << endl;
    return 0;
}
