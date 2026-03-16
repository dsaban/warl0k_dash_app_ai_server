// // Gated RNN in C++ with BPTT – Deterministic Precision Mode for WARL0K Auth
// #include <iostream>
// #include <vector>
// #include <string>
// #include <random>
// #include <cmath>
// #include <algorithm>
// #include <fstream>
// #include <ctime>
// using namespace std;

// const double LR = 0.01;

// // Activation functions
// double tanh_(double x) { return tanh(x); }
// double dtanh(double x) { double t = tanh_(x); return 1 - t * t; }

// // Random helpers
// double randn() {
//     static random_device rd;
//     static mt19937 gen(rd());
//     static normal_distribution<> d(0.0, 1.0);
//     return d(gen);
// }
// int argmax(const vector<double>& v) {
//     return distance(v.begin(), max_element(v.begin(), v.end()));
// }
// vector<double> softmax(const vector<double>& x) {
//     double maxVal = *max_element(x.begin(), x.end());
//     vector<double> exps(x.size());
//     double sum = 0;
//     for (size_t i = 0; i < x.size(); ++i) {
//         exps[i] = exp(x[i] - maxVal);
//         sum += exps[i];
//     }
//     for (double& val : exps) val /= sum;
//     return exps;
// }

// struct RNN {
//     int input_dim, hidden_dim, output_dim;
//     vector<vector<double>> Wxh, Whh, Why;
//     vector<double> bh, by;

//     RNN(int in_d, int h_d, int out_d) : input_dim(in_d), hidden_dim(h_d), output_dim(out_d) {
//         Wxh.resize(hidden_dim, vector<double>(input_dim));
//         Whh.resize(hidden_dim, vector<double>(hidden_dim));
//         Why.resize(output_dim, vector<double>(hidden_dim));
//         bh.resize(hidden_dim);
//         by.resize(output_dim);
//         for (auto& row : Wxh) for (double& v : row) v = randn() * 0.1;
//         for (auto& row : Whh) for (double& v : row) v = randn() * 0.1;
//         for (auto& row : Why) for (double& v : row) v = randn() * 0.1;
//     }

//     double forward_backward(const vector<vector<double>>& input_seq, const vector<int>& target_seq) {
//         size_t T = input_seq.size();
//         vector<vector<double>> hs(T + 1, vector<double>(hidden_dim));
//         vector<vector<double>> ys(T);
//         vector<vector<double>> probs(T);
//         double total_loss = 0.0;

//         for (size_t t = 0; t < T; ++t) {
//             for (int i = 0; i < hidden_dim; ++i) {
//                 hs[t+1][i] = bh[i];
//                 for (int j = 0; j < input_dim; ++j) hs[t+1][i] += Wxh[i][j] * input_seq[t][j];
//                 for (int j = 0; j < hidden_dim; ++j) hs[t+1][i] += Whh[i][j] * hs[t][j];
//                 hs[t+1][i] = tanh_(hs[t+1][i]);
//             }
//             ys[t].resize(output_dim);
//             for (int i = 0; i < output_dim; ++i) {
//                 ys[t][i] = by[i];
//                 for (int j = 0; j < hidden_dim; ++j) ys[t][i] += Why[i][j] * hs[t+1][j];
//             }
//             probs[t] = softmax(ys[t]);
//             total_loss -= log(probs[t][target_seq[t]] + 1e-9);
//         }

//         vector<vector<double>> dWxh(hidden_dim, vector<double>(input_dim));
//         vector<vector<double>> dWhh(hidden_dim, vector<double>(hidden_dim));
//         vector<vector<double>> dWhy(output_dim, vector<double>(hidden_dim));
//         vector<double> dbh(hidden_dim), dby(output_dim);
//         vector<double> dh_next(hidden_dim);

//         for (int t = T - 1; t >= 0; --t) {
//             vector<double> dy = probs[t];
//             dy[target_seq[t]] -= 1.0;
//             for (int i = 0; i < output_dim; ++i) {
//                 dby[i] += dy[i];
//                 for (int j = 0; j < hidden_dim; ++j)
//                     dWhy[i][j] += dy[i] * hs[t+1][j];
//             }
//             vector<double> dh(hidden_dim);
//             for (int j = 0; j < hidden_dim; ++j) {
//                 for (int i = 0; i < output_dim; ++i)
//                     dh[j] += Why[i][j] * dy[i];
//                 dh[j] += dh_next[j];
//                 double dt = dtanh(hs[t+1][j]) * dh[j];
//                 dbh[j] += dt;
//                 for (int i = 0; i < input_dim; ++i)
//                     dWxh[j][i] += dt * input_seq[t][i];
//                 for (int i = 0; i < hidden_dim; ++i)
//                     dWhh[j][i] += dt * hs[t][i];
//                 for (int i = 0; i < hidden_dim; ++i)
//                     dh_next[i] = Whh[i][j] * dt;
//             }
//         }

//         for (int i = 0; i < hidden_dim; ++i) {
//             bh[i] -= LR * dbh[i];
//             for (int j = 0; j < input_dim; ++j)
//                 Wxh[i][j] -= LR * dWxh[i][j];
//             for (int j = 0; j < hidden_dim; ++j)
//                 Whh[i][j] -= LR * dWhh[i][j];
//         }
//         for (int i = 0; i < output_dim; ++i) {
//             by[i] -= LR * dby[i];
//             for (int j = 0; j < hidden_dim; ++j)
//                 Why[i][j] -= LR * dWhy[i][j];
//         }

//         return total_loss;
//     }

//     string predict(const vector<vector<double>>& input_seq, const string& vocab) {
//         vector<double> h(hidden_dim);
//         string result = "";
//         for (auto& x : input_seq) {
//             vector<double> h_new(hidden_dim);
//             for (int i = 0; i < hidden_dim; ++i) {
//                 h_new[i] = bh[i];
//                 for (int j = 0; j < input_dim; ++j) h_new[i] += Wxh[i][j] * x[j];
//                 for (int j = 0; j < hidden_dim; ++j) h_new[i] += Whh[i][j] * h[j];
//                 h_new[i] = tanh_(h_new[i]);
//             }
//             h = h_new;

//             vector<double> logits(output_dim);
//             for (int i = 0; i < output_dim; ++i) {
//                 logits[i] = by[i];
//                 for (int j = 0; j < hidden_dim; ++j)
//                     logits[i] += Why[i][j] * h[j];
//             }
//             int idx = argmax(softmax(logits));
//             result += vocab[idx];
//         }
//         return result;
//     }
// //    save and load model for fast inference
//     void save_model(const string& filename) {
//         ofstream ofs(filename, ios::binary);
//         ofs.write(reinterpret_cast<const char*>(&input_dim), sizeof(input_dim));
//         ofs.write(reinterpret_cast<const char*>(&hidden_dim), sizeof(hidden_dim));
//         ofs.write(reinterpret_cast<const char*>(&output_dim), sizeof(output_dim));
//         for (const auto& row : Wxh) ofs.write(reinterpret_cast<const char*>(row.data()), row.size() * sizeof(double));
//         for (const auto& row : Whh) ofs.write(reinterpret_cast<const char*>(row.data()), row.size() * sizeof(double));
//         for (const auto& row : Why) ofs.write(reinterpret_cast<const char*>(row.data()), row.size() * sizeof(double));
//         ofs.write(reinterpret_cast<const char*>(bh.data()), bh.size() * sizeof(double));
//         ofs.write(reinterpret_cast<const char*>(by.data()), by.size() * sizeof(double));
//     }

//     void load_model(const string& filename) {
//         ifstream ifs(filename, ios::binary);
//         ifs.read(reinterpret_cast<char*>(&input_dim), sizeof(input_dim));
//         ifs.read(reinterpret_cast<char*>(&hidden_dim), sizeof(hidden_dim));
//         ifs.read(reinterpret_cast<char*>(&output_dim), sizeof(output_dim));
//         Wxh.resize(hidden_dim, vector<double>(input_dim));
//         Whh.resize(hidden_dim, vector<double>(hidden_dim));
//         Why.resize(output_dim, vector<double>(hidden_dim));
//         bh.resize(hidden_dim);
//         by.resize(output_dim);
//         for (auto& row : Wxh) ifs.read(reinterpret_cast<char*>(row.data()), row.size() * sizeof(double));
//         for (auto& row : Whh) ifs.read(reinterpret_cast<char*>(row.data()), row.size() * sizeof(double));
//         for (auto& row : Why) ifs.read(reinterpret_cast<char*>(row.data()), row.size() * sizeof(double));
//         ifs.read(reinterpret_cast<char*>(bh.data()), bh.size() * sizeof(double));
//         ifs.read(reinterpret_cast<char*>(by.data()), by.size() * sizeof(double));
//     }

// };

// int main() {
//     string vocab = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz1234567890!@#$%^&*()-_=+[]{}|;:',.<>?/";
//     string target = "";
//     // int seed = 42;
//     int secret_size = 32;
// //    srand(seed);
//     srand(time(0));
//     clock_t start_c = clock();
//     // clock_t end_c = clock();
//     // Generate a random target string
//     // for (int i = 0; i < secret_size; ++i) target += vocab[rand() % vocab.size()];

//     for (int i = 0; i < secret_size; ++i) target += vocab[rand() % vocab.size()];
//     cout << "[Target]: " << target << endl;

//     int input_dim = vocab.size();
//     vector<vector<double>> x_seq;
//     vector<int> t_seq;
//     for (char c : target) {
//         vector<double> onehot(input_dim, 0);
//         onehot[vocab.find(c)] = 1.0;
//         x_seq.push_back(onehot);
//         t_seq.push_back(vocab.find(c));
//     }

//     RNN net(input_dim, 32, input_dim);
//     for (int epoch = 0; epoch <= 141; ++epoch) {
//         double loss = net.forward_backward(x_seq, t_seq);
//         if (epoch % 10 == 0)
//             cout << "[Epoch " << epoch << "] Loss: " << loss << " | Reconstructed: " << net.predict(x_seq, vocab) << endl;
//     }

//     cout << "\n[Target]: " << target << endl;
//     //  check is the target string is equal to the reconstructed string
//     if (target == net.predict(x_seq, vocab))
//         // clock_t end_c = clock();
//         // double elapsed_secs_c = double(end_c - start_c) / CLOCKS_PER_SEC;
//         // cout << "[Time]: " << elapsed_secs_c * 1000 << "ms" << endl;

//         cout << "[Success]: Target string is equal to the reconstructed string" << endl;

//         // cout << "[Reconstructed]: " << net.predict(x_seq, vocab) << endl;
//         // net.save_model("model.bin");
//         // net.load_model("model.bin");
//     else

//         cout << "[Failure]: Target string is not equal to the reconstructed string" << endl;

//     clock_t end_c = clock();
//     double elapsed_secs_c = double(end_c - start_c) / CLOCKS_PER_SEC;
//     cout << "[Time train]: " << elapsed_secs_c * 1000 << "ms" << endl;


//     // calculate the estimated AI RAM usage
//     size_t ram_bytes = input_dim * 32 * sizeof(double) + 32 * 32 * sizeof(double) + input_dim * 32 * sizeof(double) + 32 * sizeof(double) + input_dim * sizeof(double);
//     cout << "[Estimated AI RAM Usage]: " << (ram_bytes / 1024.0) << " KB" << endl;

//     // save the model
//     net.save_model("model.bin");
//     // load the model
//     //  take time
//     clock_t start = clock();
//     net.load_model("model.bin");
//     // predict the target string with model.bin
//     cout << "[Reconstructed]: " << net.predict(x_seq, vocab) << endl;
//     clock_t end = clock();
//     double elapsed_secs = double(end - start) / CLOCKS_PER_SEC;
//     cout << "[Time]: " << elapsed_secs * 1000 << "ms" << endl;

//     return 0;
// }
// WARL0K PIM demo:
// Core: Gated RNN-style autoencoder for deterministic reconstruction
// Task: obfuscated_secret -> master_secret with PIM-style behavior monitoring

// #include <iostream>
// #include <vector>
// #include <string>
// #include <random>
// #include <cmath>
// #include <algorithm>
// #include <fstream>
// #include <ctime>
// #include <iomanip>

// using namespace std;

// const double LR = 0.001;

// // ----------------------
// // Activation & helpers
// // ----------------------
// double tanh_(double x) { return tanh(x); }

// // derivative when input is already tanh(z)
// double dtanh_from_h(double h) { return 1.0 - h * h; }

// double randn() {
//     static random_device rd;
//     static mt19937 gen(rd());
//     static normal_distribution<> d(0.0, 1.0);
//     return d(gen);
// }

// int argmax(const vector<double>& v) {
//     return int(distance(v.begin(), max_element(v.begin(), v.end())));
// }

// vector<double> softmax(const vector<double>& x) {
//     double maxVal = *max_element(x.begin(), x.end());
//     vector<double> exps(x.size());
//     double sum = 0.0;
//     for (size_t i = 0; i < x.size(); ++i) {
//         exps[i] = exp(x[i] - maxVal);
//         sum += exps[i];
//     }
//     if (sum <= 0) sum = 1e-12;
//     for (double& val : exps) val /= sum;
//     return exps;
// }

// // ----------------------
// // Obfuscator: master -> obf (within same vocab)
// // ----------------------
// struct Obfuscator {
//     string vocab;
//     explicit Obfuscator(const string& v) : vocab(v) {}

//     // deterministic per (seed, counter)
//     string obfuscate(const string& master, int seed, int counter) const {
//         int V = (int)vocab.size();
//         string obf;
//         obf.reserve(master.size());
//         for (size_t i = 0; i < master.size(); ++i) {
//             char c = master[i];
//             int idx = (int)vocab.find(c);
//             if (idx < 0) idx = 0; // fallback if not found

//             int mask = (seed * 31 + counter * 17 + int(i) * 7) % V;
//             int obf_idx = (idx + mask) % V;
//             obf.push_back(vocab[obf_idx]);
//         }
//         return obf;
//     }
// };

// // ----------------------
// // RNN core (with save/load)
// // ----------------------
// struct RNN {
//     int input_dim, hidden_dim, output_dim;
//     vector<vector<double>> Wxh, Whh, Why;
//     vector<double> bh, by;

//     RNN(int in_d, int h_d, int out_d)
//         : input_dim(in_d), hidden_dim(h_d), output_dim(out_d)
//     {
//         Wxh.resize(hidden_dim, vector<double>(input_dim));
//         Whh.resize(hidden_dim, vector<double>(hidden_dim));
//         Why.resize(output_dim, vector<double>(hidden_dim));
//         bh.resize(hidden_dim);
//         by.resize(output_dim);

//         for (auto& row : Wxh) for (double& v : row) v = randn() * 0.1;
//         for (auto& row : Whh) for (double& v : row) v = randn() * 0.1;
//         for (auto& row : Why) for (double& v : row) v = randn() * 0.1;
//         for (double& v : bh) v = 0.0;
//         for (double& v : by) v = 0.0;
//     }

//     // Forward + backward (BPTT)
//     double forward_backward(const vector<vector<double>>& input_seq,
//                             const vector<int>& target_seq) {
//         size_t T = input_seq.size();
//         // hs[t] = hidden state at time t (t=0 is initial zeros)
//         vector<vector<double>> hs(T + 1, vector<double>(hidden_dim, 0.0));
//         vector<vector<double>> logits(T, vector<double>(output_dim, 0.0));
//         vector<vector<double>> probs(T, vector<double>(output_dim, 0.0));

//         double total_loss = 0.0;

//         // ----- Forward -----
//         for (size_t t = 0; t < T; ++t) {
//             // h_{t+1}
//             for (int i = 0; i < hidden_dim; ++i) {
//                 double a = bh[i];
//                 for (int j = 0; j < input_dim; ++j)
//                     a += Wxh[i][j] * input_seq[t][j];
//                 for (int j = 0; j < hidden_dim; ++j)
//                     a += Whh[i][j] * hs[t][j];
//                 hs[t+1][i] = tanh_(a);
//             }
//             // y_t
//             for (int i = 0; i < output_dim; ++i) {
//                 double s = by[i];
//                 for (int j = 0; j < hidden_dim; ++j)
//                     s += Why[i][j] * hs[t+1][j];
//                 logits[t][i] = s;
//             }
//             probs[t] = softmax(logits[t]);
//             total_loss -= log(probs[t][target_seq[t]] + 1e-9);
//         }

//         // ----- Backward -----
//         vector<vector<double>> dWxh(hidden_dim, vector<double>(input_dim, 0.0));
//         vector<vector<double>> dWhh(hidden_dim, vector<double>(hidden_dim, 0.0));
//         vector<vector<double>> dWhy(output_dim, vector<double>(hidden_dim, 0.0));
//         vector<double> dbh(hidden_dim, 0.0), dby(output_dim, 0.0);
//         vector<double> dh_next(hidden_dim, 0.0);

//         for (int t = int(T) - 1; t >= 0; --t) {
//             // dy
//             vector<double> dy = probs[t];
//             dy[target_seq[t]] -= 1.0;

//             // dWhy, dby, dh from output
//             for (int i = 0; i < output_dim; ++i) {
//                 dby[i] += dy[i];
//                 for (int j = 0; j < hidden_dim; ++j)
//                     dWhy[i][j] += dy[i] * hs[t+1][j];
//             }

//             vector<double> dh(hidden_dim, 0.0);
//             for (int j = 0; j < hidden_dim; ++j) {
//                 double sum = 0.0;
//                 for (int i = 0; i < output_dim; ++i)
//                     sum += Why[i][j] * dy[i];
//                 dh[j] = sum + dh_next[j];
//             }

//             vector<double> da(hidden_dim, 0.0);
//             for (int j = 0; j < hidden_dim; ++j) {
//                 da[j] = dh[j] * dtanh_from_h(hs[t+1][j]);
//                 dbh[j] += da[j];
//                 for (int i = 0; i < input_dim; ++i)
//                     dWxh[j][i] += da[j] * input_seq[t][i];
//                 for (int i = 0; i < hidden_dim; ++i)
//                     dWhh[j][i] += da[j] * hs[t][i];
//             }

//             // dh_next = Whh^T * da
//             fill(dh_next.begin(), dh_next.end(), 0.0);
//             for (int i = 0; i < hidden_dim; ++i) {
//                 double sum = 0.0;
//                 for (int j = 0; j < hidden_dim; ++j)
//                     sum += Whh[j][i] * da[j];
//                 dh_next[i] = sum;
//             }
//         }

//         // gradient step
//         for (int i = 0; i < hidden_dim; ++i) {
//             bh[i] -= LR * dbh[i];
//             for (int j = 0; j < input_dim; ++j)
//                 Wxh[i][j] -= LR * dWxh[i][j];
//             for (int j = 0; j < hidden_dim; ++j)
//                 Whh[i][j] -= LR * dWhh[i][j];
//         }
//         for (int i = 0; i < output_dim; ++i) {
//             by[i] -= LR * dby[i];
//             for (int j = 0; j < hidden_dim; ++j)
//                 Why[i][j] -= LR * dWhy[i][j];
//         }

//         return total_loss;
//     }

//     // Predict sequence (greedy argmax per step)
//     string predict(const vector<vector<double>>& input_seq, const string& vocab) {
//         vector<double> h(hidden_dim, 0.0);
//         string result;
//         for (const auto& x : input_seq) {
//             vector<double> h_new(hidden_dim, 0.0);
//             for (int i = 0; i < hidden_dim; ++i) {
//                 double a = bh[i];
//                 for (int j = 0; j < input_dim; ++j) a += Wxh[i][j] * x[j];
//                 for (int j = 0; j < hidden_dim; ++j) a += Whh[i][j] * h[j];
//                 h_new[i] = tanh_(a);
//             }
//             h = h_new;

//             vector<double> logits(output_dim, 0.0);
//             for (int i = 0; i < output_dim; ++i) {
//                 double s = by[i];
//                 for (int j = 0; j < hidden_dim; ++j)
//                     s += Why[i][j] * h[j];
//                 logits[i] = s;
//             }
//             vector<double> p = softmax(logits);
//             int idx = argmax(p);
//             result.push_back(vocab[idx]);
//         }
//         return result;
//     }

//     // save and load model for fast inference
//     void save_model(const string& filename) {
//         ofstream ofs(filename, ios::binary);
//         ofs.write(reinterpret_cast<const char*>(&input_dim), sizeof(input_dim));
//         ofs.write(reinterpret_cast<const char*>(&hidden_dim), sizeof(hidden_dim));
//         ofs.write(reinterpret_cast<const char*>(&output_dim), sizeof(output_dim));
//         for (const auto& row : Wxh)
//             ofs.write(reinterpret_cast<const char*>(row.data()), row.size() * sizeof(double));
//         for (const auto& row : Whh)
//             ofs.write(reinterpret_cast<const char*>(row.data()), row.size() * sizeof(double));
//         for (const auto& row : Why)
//             ofs.write(reinterpret_cast<const char*>(row.data()), row.size() * sizeof(double));
//         ofs.write(reinterpret_cast<const char*>(bh.data()), bh.size() * sizeof(double));
//         ofs.write(reinterpret_cast<const char*>(by.data()), by.size() * sizeof(double));
//     }

//     void load_model(const string& filename) {
//         ifstream ifs(filename, ios::binary);
//         if (!ifs) {
//             cerr << "[Error] Could not open model file: " << filename << "\n";
//             return;
//         }
//         ifs.read(reinterpret_cast<char*>(&input_dim), sizeof(input_dim));
//         ifs.read(reinterpret_cast<char*>(&hidden_dim), sizeof(hidden_dim));
//         ifs.read(reinterpret_cast<char*>(&output_dim), sizeof(output_dim));
//         Wxh.assign(hidden_dim, vector<double>(input_dim));
//         Whh.assign(hidden_dim, vector<double>(hidden_dim));
//         Why.assign(output_dim, vector<double>(hidden_dim));
//         bh.assign(hidden_dim, 0.0);
//         by.assign(output_dim, 0.0);
//         for (auto& row : Wxh)
//             ifs.read(reinterpret_cast<char*>(row.data()), row.size() * sizeof(double));
//         for (auto& row : Whh)
//             ifs.read(reinterpret_cast<char*>(row.data()), row.size() * sizeof(double));
//         for (auto& row : Why)
//             ifs.read(reinterpret_cast<char*>(row.data()), row.size() * sizeof(double));
//         ifs.read(reinterpret_cast<char*>(bh.data()), bh.size() * sizeof(double));
//         ifs.read(reinterpret_cast<char*>(by.data()), by.size() * sizeof(double));
//     }

//     // estimated parameter memory (in bytes)
//     size_t param_bytes() const {
//         size_t sz = 0;
//         sz += (size_t)input_dim * hidden_dim * sizeof(double);   // Wxh
//         sz += (size_t)hidden_dim * hidden_dim * sizeof(double);  // Whh
//         sz += (size_t)output_dim * hidden_dim * sizeof(double);  // Why
//         sz += (size_t)hidden_dim * sizeof(double);               // bh
//         sz += (size_t)output_dim * sizeof(double);               // by
//         return sz;
//     }
// };

// // ----------------------
// // Build input seq from obf + (seed,counter)
// // ----------------------
// vector<vector<double>> build_input_seq(const string& obf,
//                                        const string& vocab,
//                                        int seed,
//                                        int counter) {
//     int V = (int)vocab.size();
//     int input_dim = V + 2;
//     vector<vector<double>> x_seq;
//     x_seq.reserve(obf.size());

//     double s_norm = seed / 16.0;    // arbitrary normalization for demo
//     double c_norm = counter / 64.0;

//     for (char c : obf) {
//         vector<double> x(input_dim, 0.0);
//         int idx = (int)vocab.find(c);
//         if (idx < 0) idx = 0;
//         x[idx] = 1.0;
//         x[V]   = s_norm;
//         x[V+1] = c_norm;
//         x_seq.push_back(x);
//     }
//     return x_seq;
// }

// // ----------------------
// // PIM chain monitor
// // ----------------------
// void pim_chain_monitor(RNN& net,
//                        const string& master,
//                        const string& vocab,
//                        const Obfuscator& obf,
//                        int seed,
//                        int start_counter,
//                        int length,
//                        int window_size,
//                        double err_threshold) {
//     int L = (int)master.size();
//     int window_start = start_counter;
//     int window_end   = start_counter + window_size - 1;

//     cout << "\n=== PIM CHAIN MONITOR ===\n";
//     cout << "seed=" << seed << " window=[" << window_start << "," << window_end
//          << "] length=" << length << " err_threshold=" << err_threshold << "\n";

//     for (int i = 0; i < length; ++i) {
//         int counter = start_counter + i;
//         string obf_str = obf.obfuscate(master, seed, counter);
//         auto x_seq = build_input_seq(obf_str, vocab, seed, counter);
//         string recon = net.predict(x_seq, vocab);

//         int mismatches = 0;
//         for (int t = 0; t < L; ++t) {
//             if (recon[t] != master[t]) mismatches++;
//         }
//         double err_rate = (double)mismatches / (double)L;
//         bool inside_window = (counter >= window_start && counter <= window_end);
//         bool ok = inside_window && (err_rate <= err_threshold);

//         string status;
//         if (!inside_window) status = "OUT_OF_WINDOW";
//         else if (!ok)       status = "ANOMALY";
//         else                status = "OK";

//         cout << " ctr=" << setw(3) << counter
//              << " err_rate=" << fixed << setprecision(3) << err_rate
//              << " inside_window=" << (inside_window ? "true " : "false")
//              << " status=" << status << "\n";
//     }
// }

// // ----------------------
// // MAIN: PIM demo
// // ----------------------
// int main() {
//     ios::sync_with_stdio(false);
//     cin.tie(nullptr);

//     // Vocabulary and random master secret
//     string vocab = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz1234567890!@#$%^&*()-_=+[]{}|;:',.<>?/";
//     int vocab_size = (int)vocab.size();
//     int secret_size = 32;

//     srand((unsigned)time(nullptr));
//     string master;
//     master.reserve(secret_size);
//     for (int i = 0; i < secret_size; ++i)
//         master.push_back(vocab[rand() % vocab_size]);

//     cout << "[Master secret]: " << master << "\n";

//     // Obfuscator
//     Obfuscator obf(vocab);

//     // RNN: input_dim = vocab + 2 (seed, counter), output_dim = vocab
//     int hidden_dim = 32;
//     int input_dim = vocab_size + 2;
//     int output_dim = vocab_size;
//     RNN net(input_dim, hidden_dim, output_dim);

//     // Print model size
//     size_t bytes = net.param_bytes();
//     cout << "[Model params]: " << bytes << " bytes ("
//          << fixed << setprecision(2) << (bytes / 1024.0) << " KB)\n";

//     // Build target indices once (master chars)
//     vector<int> target_indices(secret_size);
//     for (int i = 0; i < secret_size; ++i) {
//         int idx = (int)vocab.find(master[i]);
//         if (idx < 0) idx = 0;
//         target_indices[i] = idx;
//     }

//     // Training hyperparameters
//     int epochs = 500;
//     pair<int,int> seed_range    = {1, 3};
//     pair<int,int> counter_range = {1, 8};

//     cout << "\n=== TRAINING (obf -> master) ===\n";
//     clock_t train_start = clock();

//     for (int epoch = 0; epoch <= epochs; ++epoch) {
//         double total_loss = 0.0;
//         int num_pairs = 0;

//         for (int s = seed_range.first; s <= seed_range.second; ++s) {
//             for (int c = counter_range.first; c <= counter_range.second; ++c) {
//                 string obf_str = obf.obfuscate(master, s, c);
//                 auto x_seq = build_input_seq(obf_str, vocab, s, c);
//                 double loss = net.forward_backward(x_seq, target_indices);
//                 total_loss += loss;
//                 num_pairs++;
//             }
//         }

//         double avg_loss = total_loss / (double)num_pairs;
//         if (epoch % 10 == 0)
//             cout << "[Epoch " << setw(3) << epoch << "] avg_loss=" << avg_loss << "\n";
//     }

//     clock_t train_end = clock();
//     double train_ms = double(train_end - train_start) / CLOCKS_PER_SEC * 1000.0;
//     cout << "[Time train]: " << fixed << setprecision(3) << train_ms << " ms\n";

//     // Test reconstruction for one (seed, counter)
//     int test_seed = 2;
//     int test_counter = 5;
//     string obf_test = obf.obfuscate(master, test_seed, test_counter);
//     auto x_test = build_input_seq(obf_test, vocab, test_seed, test_counter);
//     string recon = net.predict(x_test, vocab);

//     cout << "\n=== ROUNDTRIP TEST ===\n";
//     cout << "Master       : " << master << "\n";
//     cout << "Obfuscated   : " << obf_test << "\n";
//     cout << "Reconstructed: " << recon << "\n";
//     cout << "Equal?       : " << ((recon == master) ? "true" : "false") << "\n";

//     // Save model
//     net.save_model("model.bin");

//     // Load model into fresh instance and time fast inference
//     clock_t load_infer_start = clock();
//     RNN net_loaded(1,1,1);  // dummy, will be overwritten
//     net_loaded.load_model("model.bin");
//     auto x_test2 = build_input_seq(obf_test, vocab, test_seed, test_counter);
//     string recon2 = net_loaded.predict(x_test2, vocab);
//     clock_t load_infer_end = clock();
//     double load_infer_ms = double(load_infer_end - load_infer_start) / CLOCKS_PER_SEC * 1000.0;

//     cout << "\n=== FAST INFERENCE (load + predict) ===\n";
//     cout << "Reconstructed (loaded): " << recon2 << "\n";
//     cout << "Equal?                 : " << ((recon2 == master) ? "true" : "false") << "\n";
//     cout << "Time (load+infer)      : " << fixed << setprecision(3) << load_infer_ms << " ms\n";

//     // PIM behavior analysis over drifting window of counters
//     pim_chain_monitor(net_loaded,
//                       master,
//                       vocab,
//                       obf,
//                       /*seed*/ test_seed,
//                       /*start_counter*/ 1,
//                       /*length*/ 16,
//                       /*window_size*/ 8,   // legal counters: 1..8
//                       /*err_threshold*/ 0.25);

//     return 0;
// }
// WARL0K PIM demo:
// Core: Gated RNN-style autoencoder for deterministic reconstruction
// Task: obfuscated_secret -> master_secret with PIM-style behavior monitoring

// #include <iostream>
// #include <vector>
// #include <string>
// #include <random>
// #include <cmath>
// #include <algorithm>
// #include <fstream>
// #include <ctime>
// #include <iomanip>

// using namespace std;

// const double LR = 0.001;

// // ----------------------
// // Activation & helpers
// // ----------------------
// double tanh_(double x) { return tanh(x); }

// // derivative when input is already tanh(z)
// double dtanh_from_h(double h) { return 1.0 - h * h; }

// double randn() {
//     static random_device rd;
//     static mt19937 gen(rd());
//     static normal_distribution<> d(0.0, 1.0);
//     return d(gen);
// }

// int argmax(const vector<double>& v) {
//     return int(distance(v.begin(), max_element(v.begin(), v.end())));
// }

// vector<double> softmax(const vector<double>& x) {
//     double maxVal = *max_element(x.begin(), x.end());
//     vector<double> exps(x.size());
//     double sum = 0.0;
//     for (size_t i = 0; i < x.size(); ++i) {
//         exps[i] = exp(x[i] - maxVal);
//         sum += exps[i];
//     }
//     if (sum <= 0) sum = 1e-12;
//     for (double& val : exps) val /= sum;
//     return exps;
// }

// // ----------------------
// // Obfuscator: master -> obf (within same vocab)
// // ----------------------
// struct Obfuscator {
//     string vocab;
//     explicit Obfuscator(const string& v) : vocab(v) {}

//     // deterministic per (seed, counter)
//     string obfuscate(const string& master, int seed, int counter) const {
//         int V = (int)vocab.size();
//         string obf;
//         obf.reserve(master.size());
//         for (size_t i = 0; i < master.size(); ++i) {
//             char c = master[i];
//             int idx = (int)vocab.find(c);
//             if (idx < 0) idx = 0; // fallback if not found

//             int mask = (seed * 31 + counter * 17 + int(i) * 7) % V;
//             int obf_idx = (idx + mask) % V;
//             obf.push_back(vocab[obf_idx]);
//         }
//         return obf;
//     }
// };

// // ----------------------
// // RNN core (with save/load)
// // ----------------------
// struct RNN {
//     int input_dim, hidden_dim, output_dim;
//     vector<vector<double>> Wxh, Whh, Why;
//     vector<double> bh, by;

//     RNN(int in_d, int h_d, int out_d)
//         : input_dim(in_d), hidden_dim(h_d), output_dim(out_d)
//     {
//         Wxh.resize(hidden_dim, vector<double>(input_dim));
//         Whh.resize(hidden_dim, vector<double>(hidden_dim));
//         Why.resize(output_dim, vector<double>(hidden_dim));
//         bh.resize(hidden_dim);
//         by.resize(output_dim);

//         for (auto& row : Wxh) for (double& v : row) v = randn() * 0.1;
//         for (auto& row : Whh) for (double& v : row) v = randn() * 0.1;
//         for (auto& row : Why) for (double& v : row) v = randn() * 0.1;
//         for (double& v : bh) v = 0.0;
//         for (double& v : by) v = 0.0;
//     }

//     // Forward + backward (BPTT)
//     double forward_backward(const vector<vector<double>>& input_seq,
//                             const vector<int>& target_seq) {
//         size_t T = input_seq.size();
//         // hs[t] = hidden state at time t (t=0 is initial zeros)
//         vector<vector<double>> hs(T + 1, vector<double>(hidden_dim, 0.0));
//         vector<vector<double>> logits(T, vector<double>(output_dim, 0.0));
//         vector<vector<double>> probs(T, vector<double>(output_dim, 0.0));

//         double total_loss = 0.0;

//         // ----- Forward -----
//         for (size_t t = 0; t < T; ++t) {
//             // h_{t+1}
//             for (int i = 0; i < hidden_dim; ++i) {
//                 double a = bh[i];
//                 for (int j = 0; j < input_dim; ++j)
//                     a += Wxh[i][j] * input_seq[t][j];
//                 for (int j = 0; j < hidden_dim; ++j)
//                     a += Whh[i][j] * hs[t][j];
//                 hs[t+1][i] = tanh_(a);
//             }
//             // y_t
//             for (int i = 0; i < output_dim; ++i) {
//                 double s = by[i];
//                 for (int j = 0; j < hidden_dim; ++j)
//                     s += Why[i][j] * hs[t+1][j];
//                 logits[t][i] = s;
//             }
//             probs[t] = softmax(logits[t]);
//             total_loss -= log(probs[t][target_seq[t]] + 1e-9);
//         }

//         // ----- Backward -----
//         vector<vector<double>> dWxh(hidden_dim, vector<double>(input_dim, 0.0));
//         vector<vector<double>> dWhh(hidden_dim, vector<double>(hidden_dim, 0.0));
//         vector<vector<double>> dWhy(output_dim, vector<double>(hidden_dim, 0.0));
//         vector<double> dbh(hidden_dim, 0.0), dby(output_dim, 0.0);
//         vector<double> dh_next(hidden_dim, 0.0);

//         for (int t = int(T) - 1; t >= 0; --t) {
//             // dy
//             vector<double> dy = probs[t];
//             dy[target_seq[t]] -= 1.0;

//             // dWhy, dby, dh from output
//             for (int i = 0; i < output_dim; ++i) {
//                 dby[i] += dy[i];
//                 for (int j = 0; j < hidden_dim; ++j)
//                     dWhy[i][j] += dy[i] * hs[t+1][j];
//             }

//             vector<double> dh(hidden_dim, 0.0);
//             for (int j = 0; j < hidden_dim; ++j) {
//                 double sum = 0.0;
//                 for (int i = 0; i < output_dim; ++i)
//                     sum += Why[i][j] * dy[i];
//                 dh[j] = sum + dh_next[j];
//             }

//             vector<double> da(hidden_dim, 0.0);
//             for (int j = 0; j < hidden_dim; ++j) {
//                 da[j] = dh[j] * dtanh_from_h(hs[t+1][j]);
//                 dbh[j] += da[j];
//                 for (int i = 0; i < input_dim; ++i)
//                     dWxh[j][i] += da[j] * input_seq[t][i];
//                 for (int i = 0; i < hidden_dim; ++i)
//                     dWhh[j][i] += da[j] * hs[t][i];
//             }

//             // dh_next = Whh^T * da
//             fill(dh_next.begin(), dh_next.end(), 0.0);
//             for (int i = 0; i < hidden_dim; ++i) {
//                 double sum = 0.0;
//                 for (int j = 0; j < hidden_dim; ++j)
//                     sum += Whh[j][i] * da[j];
//                 dh_next[i] = sum;
//             }
//         }

//         // gradient step
//         for (int i = 0; i < hidden_dim; ++i) {
//             bh[i] -= LR * dbh[i];
//             for (int j = 0; j < input_dim; ++j)
//                 Wxh[i][j] -= LR * dWxh[i][j];
//             for (int j = 0; j < hidden_dim; ++j)
//                 Whh[i][j] -= LR * dWhh[i][j];
//         }
//         for (int i = 0; i < output_dim; ++i) {
//             by[i] -= LR * dby[i];
//             for (int j = 0; j < hidden_dim; ++j)
//                 Why[i][j] -= LR * dWhy[i][j];
//         }

//         return total_loss;
//     }

//     // Predict sequence (greedy argmax per step)
//     string predict(const vector<vector<double>>& input_seq, const string& vocab) {
//         vector<double> h(hidden_dim, 0.0);
//         string result;
//         for (const auto& x : input_seq) {
//             vector<double> h_new(hidden_dim, 0.0);
//             for (int i = 0; i < hidden_dim; ++i) {
//                 double a = bh[i];
//                 for (int j = 0; j < input_dim; ++j) a += Wxh[i][j] * x[j];
//                 for (int j = 0; j < hidden_dim; ++j) a += Whh[i][j] * h[j];
//                 h_new[i] = tanh_(a);
//             }
//             h = h_new;

//             vector<double> logits(output_dim, 0.0);
//             for (int i = 0; i < output_dim; ++i) {
//                 double s = by[i];
//                 for (int j = 0; j < hidden_dim; ++j)
//                     s += Why[i][j] * h[j];
//                 logits[i] = s;
//             }
//             vector<double> p = softmax(logits);
//             int idx = argmax(p);
//             result.push_back(vocab[idx]);
//         }
//         return result;
//     }

//     // save and load model for fast inference
//     void save_model(const string& filename) {
//         ofstream ofs(filename, ios::binary);
//         ofs.write(reinterpret_cast<const char*>(&input_dim), sizeof(input_dim));
//         ofs.write(reinterpret_cast<const char*>(&hidden_dim), sizeof(hidden_dim));
//         ofs.write(reinterpret_cast<const char*>(&output_dim), sizeof(output_dim));
//         for (const auto& row : Wxh)
//             ofs.write(reinterpret_cast<const char*>(row.data()), row.size() * sizeof(double));
//         for (const auto& row : Whh)
//             ofs.write(reinterpret_cast<const char*>(row.data()), row.size() * sizeof(double));
//         for (const auto& row : Why)
//             ofs.write(reinterpret_cast<const char*>(row.data()), row.size() * sizeof(double));
//         ofs.write(reinterpret_cast<const char*>(bh.data()), bh.size() * sizeof(double));
//         ofs.write(reinterpret_cast<const char*>(by.data()), by.size() * sizeof(double));
//     }

//     void load_model(const string& filename) {
//         ifstream ifs(filename, ios::binary);
//         if (!ifs) {
//             cerr << "[Error] Could not open model file: " << filename << "\n";
//             return;
//         }
//         ifs.read(reinterpret_cast<char*>(&input_dim), sizeof(input_dim));
//         ifs.read(reinterpret_cast<char*>(&hidden_dim), sizeof(hidden_dim));
//         ifs.read(reinterpret_cast<char*>(&output_dim), sizeof(output_dim));
//         Wxh.assign(hidden_dim, vector<double>(input_dim));
//         Whh.assign(hidden_dim, vector<double>(hidden_dim));
//         Why.assign(output_dim, vector<double>(hidden_dim));
//         bh.assign(hidden_dim, 0.0);
//         by.assign(output_dim, 0.0);
//         for (auto& row : Wxh)
//             ifs.read(reinterpret_cast<char*>(row.data()), row.size() * sizeof(double));
//         for (auto& row : Whh)
//             ifs.read(reinterpret_cast<char*>(row.data()), row.size() * sizeof(double));
//         for (auto& row : Why)
//             ifs.read(reinterpret_cast<char*>(row.data()), row.size() * sizeof(double));
//         ifs.read(reinterpret_cast<char*>(bh.data()), bh.size() * sizeof(double));
//         ifs.read(reinterpret_cast<char*>(by.data()), by.size() * sizeof(double));
//     }

//     // estimated parameter memory (in bytes)
//     size_t param_bytes() const {
//         size_t sz = 0;
//         sz += (size_t)input_dim * hidden_dim * sizeof(double);   // Wxh
//         sz += (size_t)hidden_dim * hidden_dim * sizeof(double);  // Whh
//         sz += (size_t)output_dim * hidden_dim * sizeof(double);  // Why
//         sz += (size_t)hidden_dim * sizeof(double);               // bh
//         sz += (size_t)output_dim * sizeof(double);               // by
//         return sz;
//     }
// };

// // ----------------------
// // Build input seq from obf + (seed,counter)
// // ----------------------
// vector<vector<double>> build_input_seq(const string& obf,
//                                        const string& vocab,
//                                        int seed,
//                                        int counter) {
//     int V = (int)vocab.size();
//     int input_dim = V + 2;
//     vector<vector<double>> x_seq;
//     x_seq.reserve(obf.size());

//     double s_norm = seed / 16.0;    // arbitrary normalization for demo
//     double c_norm = counter / 64.0;

//     for (char c : obf) {
//         vector<double> x(input_dim, 0.0);
//         int idx = (int)vocab.find(c);
//         if (idx < 0) idx = 0;
//         x[idx] = 1.0;
//         x[V]   = s_norm;
//         x[V+1] = c_norm;
//         x_seq.push_back(x);
//     }
//     return x_seq;
// }

// // ----------------------
// // PIM chain monitor
// // ----------------------
// void pim_chain_monitor(RNN& net,
//                        const string& master,
//                        const string& vocab,
//                        const Obfuscator& obf,
//                        int seed,
//                        int start_counter,
//                        int length,
//                        int window_size,
//                        double err_threshold) {
//     int L = (int)master.size();
//     int window_start = start_counter;
//     int window_end   = start_counter + window_size - 1;

//     cout << "\n=== PIM CHAIN MONITOR ===\n";
//     cout << "seed=" << seed << " window=[" << window_start << "," << window_end
//          << "] length=" << length << " err_threshold=" << err_threshold << "\n";

//     for (int i = 0; i < length; ++i) {
//         int counter = start_counter + i;
//         string obf_str = obf.obfuscate(master, seed, counter);
//         auto x_seq = build_input_seq(obf_str, vocab, seed, counter);
//         string recon = net.predict(x_seq, vocab);

//         int mismatches = 0;
//         for (int t = 0; t < L; ++t) {
//             if (recon[t] != master[t]) mismatches++;
//         }
//         double err_rate = (double)mismatches / (double)L;
//         bool inside_window = (counter >= window_start && counter <= window_end);
//         bool ok = inside_window && (err_rate <= err_threshold);

//         string status;
//         if (!inside_window) status = "OUT_OF_WINDOW";
//         else if (!ok)       status = "ANOMALY";
//         else                status = "OK";

//         cout << " ctr=" << setw(3) << counter
//              << " err_rate=" << fixed << setprecision(3) << err_rate
//              << " inside_window=" << (inside_window ? "true " : "false")
//              << " status=" << status << "\n";
//     }
// }

// // ----------------------
// // MAIN: PIM demo
// // ----------------------
// int main() {
//     ios::sync_with_stdio(false);
//     cin.tie(nullptr);

//     // Vocabulary and random master secret
//     string vocab = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz1234567890!@#$%^&*()-_=+[]{}|;:',.<>?/";
//     int vocab_size = (int)vocab.size();
//     int secret_size = 32;

//     srand((unsigned)time(nullptr));
//     string master;
//     master.reserve(secret_size);
//     for (int i = 0; i < secret_size; ++i)
//         master.push_back(vocab[rand() % vocab_size]);

//     cout << "[Master secret]: " << master << "\n";

//     // Obfuscator
//     Obfuscator obf(vocab);

//     // RNN: input_dim = vocab + 2 (seed, counter), output_dim = vocab
//     int hidden_dim = 32;
//     int input_dim = vocab_size + 2;
//     int output_dim = vocab_size;
//     RNN net(input_dim, hidden_dim, output_dim);

//     // Print model size
//     size_t bytes = net.param_bytes();
//     cout << "[Model params]: " << bytes << " bytes ("
//          << fixed << setprecision(2) << (bytes / 1024.0) << " KB)\n";

//     // Build target indices once (master chars)
//     vector<int> target_indices(secret_size);
//     for (int i = 0; i < secret_size; ++i) {
//         int idx = (int)vocab.find(master[i]);
//         if (idx < 0) idx = 0;
//         target_indices[i] = idx;
//     }

//     // Training hyperparameters
//     int epochs = 300;
//     pair<int,int> seed_range    = {1, 3};
//     pair<int,int> counter_range = {1, 8};

//     cout << "\n=== TRAINING (obf -> master) ===\n";

//     clock_t train_start = clock();
//     double sum_epoch_ms = 0.0;

//     for (int epoch = 0; epoch <= epochs; ++epoch) {
//         clock_t epoch_start = clock();

//         double total_loss = 0.0;
//         int num_pairs = 0;

//         for (int s = seed_range.first; s <= seed_range.second; ++s) {
//             for (int c = counter_range.first; c <= counter_range.second; ++c) {
//                 string obf_str = obf.obfuscate(master, s, c);
//                 auto x_seq = build_input_seq(obf_str, vocab, s, c);
//                 double loss = net.forward_backward(x_seq, target_indices);
//                 total_loss += loss;
//                 num_pairs++;
//             }
//         }

//         double avg_loss = total_loss / (double)num_pairs;

//         clock_t epoch_end = clock();
//         double epoch_ms = double(epoch_end - epoch_start) / CLOCKS_PER_SEC * 1000.0;
//         sum_epoch_ms += epoch_ms;

//         if (epoch % 10 == 0) {
//             cout << "[Epoch " << setw(3) << epoch << "] avg_loss=" << avg_loss
//                  << " | epoch_time=" << fixed << setprecision(3) << epoch_ms << " ms\n";
//         }
//     }

//     clock_t train_end = clock();
//     double train_ms = double(train_end - train_start) / CLOCKS_PER_SEC * 1000.0;
//     double avg_epoch_ms = sum_epoch_ms / double(epochs + 1);

//     cout << "[Time train - total]: " << fixed << setprecision(3) << train_ms << " ms\n";
//     cout << "[Time train - avg/epoch]: " << fixed << setprecision(3) << avg_epoch_ms << " ms\n";

//     // Test reconstruction for one (seed, counter)
//     int test_seed = 4;
//     int test_counter = 5;
//     string obf_test = obf.obfuscate(master, test_seed, test_counter);
//     auto x_test = build_input_seq(obf_test, vocab, test_seed, test_counter);
//     string recon = net.predict(x_test, vocab);

//     cout << "\n=== ROUNDTRIP TEST ===\n";
//     cout << "Master       : " << master << "\n";
//     cout << "Obfuscated   : " << obf_test << "\n";
//     cout << "Reconstructed: " << recon << "\n";
//     cout << "Equal?       : " << ((recon == master) ? "true" : "false") << "\n";

//     // Save model
//     net.save_model("model.bin");

//     // Load model into fresh instance and time fast inference
//     clock_t load_infer_start = clock();
//     RNN net_loaded(1,1,1);  // dummy, overwritten by load
//     net_loaded.load_model("model.bin");
//     auto x_test2 = build_input_seq(obf_test, vocab, test_seed, test_counter);
//     string recon2 = net_loaded.predict(x_test2, vocab);
//     clock_t load_infer_end = clock();
//     double load_infer_ms = double(load_infer_end - load_infer_start) / CLOCKS_PER_SEC * 1000.0;

//     cout << "\n=== FAST INFERENCE (load + predict) ===\n";
//     cout << "Reconstructed (loaded): " << recon2 << "\n";
//     cout << "Equal?                 : " << ((recon2 == master) ? "true" : "false") << "\n";
//     cout << "Time (load+infer)      : " << fixed << setprecision(3) << load_infer_ms << " ms\n";

//     // PIM behavior analysis over drifting window of counters
//     pim_chain_monitor(net_loaded,
//                       master,
//                       vocab,
//                       obf,
//                       /*seed*/ test_seed,
//                       /*start_counter*/ 1,
//                       /*length*/ 16,
//                       /*window_size*/ 8,   // legal counters: 1..8
//                       /*err_threshold*/ 0.25);

//     return 0;
// }
// WARL0K PIM demo with seed+counter window control
// Core: RNN-style autoencoder for deterministic reconstruction
// Task: obfuscated_secret -> master_secret with PIM-style behavior monitoring

// ---------------------first PIM------------------------------------------------------------------

// #include <iostream>
// #include <vector>
// #include <string>
// #include <random>
// #include <cmath>
// #include <algorithm>
// #include <fstream>
// #include <ctime>
// #include <iomanip>

// using namespace std;

// const double LR = 0.002;

// // ----------------------
// // Activation & helpers
// // ----------------------
// double tanh_(double x) { return tanh(x); }

// // derivative when input is already tanh(z)
// double dtanh_from_h(double h) { return 1.0 - h * h; }

// double randn() {
//     static random_device rd;
//     static mt19937 gen(rd());
//     static normal_distribution<> d(0.0, 1.0);
//     return d(gen);
// }

// int argmax(const vector<double>& v) {
//     return int(distance(v.begin(), max_element(v.begin(), v.end())));
// }

// vector<double> softmax(const vector<double>& x) {
//     double maxVal = *max_element(x.begin(), x.end());
//     vector<double> exps(x.size());
//     double sum = 0.0;
//     for (size_t i = 0; i < x.size(); ++i) {
//         exps[i] = exp(x[i] - maxVal);
//         sum += exps[i];
//     }
//     if (sum <= 0) sum = 1e-12;
//     for (double& val : exps) val /= sum;
//     return exps;
// }

// // ----------------------
// // Obfuscator: master -> obf (within same vocab)
// // ----------------------
// struct Obfuscator {
//     string vocab;
//     explicit Obfuscator(const string& v) : vocab(v) {}

//     // deterministic per (seed, counter)
//     string obfuscate(const string& master, int seed, int counter) const {
//         int V = (int)vocab.size();
//         string obf;
//         obf.reserve(master.size());
//         for (size_t i = 0; i < master.size(); ++i) {
//             char c = master[i];
//             int idx = (int)vocab.find(c);
//             if (idx < 0) idx = 0; // fallback if not found

//             int mask = (seed * 31 + counter * 17 + int(i) * 7) % V;
//             int obf_idx = (idx + mask) % V;
//             obf.push_back(vocab[obf_idx]);
//         }
//         return obf;
//     }
// };

// // ----------------------
// // RNN core (with save/load)
// // ----------------------
// struct RNN {
//     int input_dim, hidden_dim, output_dim;
//     vector<vector<double>> Wxh, Whh, Why;
//     vector<double> bh, by;

//     RNN(int in_d, int h_d, int out_d)
//         : input_dim(in_d), hidden_dim(h_d), output_dim(out_d)
//     {
//         Wxh.resize(hidden_dim, vector<double>(input_dim));
//         Whh.resize(hidden_dim, vector<double>(hidden_dim));
//         Why.resize(output_dim, vector<double>(hidden_dim));
//         bh.resize(hidden_dim);
//         by.resize(output_dim);

//         for (auto& row : Wxh) for (double& v : row) v = randn() * 0.1;
//         for (auto& row : Whh) for (double& v : row) v = randn() * 0.1;
//         for (auto& row : Why) for (double& v : row) v = randn() * 0.1;
//         for (double& v : bh) v = 0.0;
//         for (double& v : by) v = 0.0;
//     }

//     // Forward + backward (BPTT)
//     double forward_backward(const vector<vector<double>>& input_seq,
//                             const vector<int>& target_seq) {
//         size_t T = input_seq.size();
//         // hs[t] = hidden state at time t (t=0 is initial zeros)
//         vector<vector<double>> hs(T + 1, vector<double>(hidden_dim, 0.0));
//         vector<vector<double>> logits(T, vector<double>(output_dim, 0.0));
//         vector<vector<double>> probs(T, vector<double>(output_dim, 0.0));

//         double total_loss = 0.0;

//         // ----- Forward -----
//         for (size_t t = 0; t < T; ++t) {
//             // h_{t+1}
//             for (int i = 0; i < hidden_dim; ++i) {
//                 double a = bh[i];
//                 for (int j = 0; j < input_dim; ++j)
//                     a += Wxh[i][j] * input_seq[t][j];
//                 for (int j = 0; j < hidden_dim; ++j)
//                     a += Whh[i][j] * hs[t][j];
//                 hs[t+1][i] = tanh_(a);
//             }
//             // y_t
//             for (int i = 0; i < output_dim; ++i) {
//                 double s = by[i];
//                 for (int j = 0; j < hidden_dim; ++j)
//                     s += Why[i][j] * hs[t+1][j];
//                 logits[t][i] = s;
//             }
//             probs[t] = softmax(logits[t]);
//             total_loss -= log(probs[t][target_seq[t]] + 1e-9);
//         }

//         // ----- Backward -----
//         vector<vector<double>> dWxh(hidden_dim, vector<double>(input_dim, 0.0));
//         vector<vector<double>> dWhh(hidden_dim, vector<double>(hidden_dim, 0.0));
//         vector<vector<double>> dWhy(output_dim, vector<double>(hidden_dim, 0.0));
//         vector<double> dbh(hidden_dim, 0.0), dby(output_dim, 0.0);
//         vector<double> dh_next(hidden_dim, 0.0);

//         for (int t = int(T) - 1; t >= 0; --t) {
//             // dy
//             vector<double> dy = probs[t];
//             dy[target_seq[t]] -= 1.0;

//             // dWhy, dby, dh from output
//             for (int i = 0; i < output_dim; ++i) {
//                 dby[i] += dy[i];
//                 for (int j = 0; j < hidden_dim; ++j)
//                     dWhy[i][j] += dy[i] * hs[t+1][j];
//             }

//             vector<double> dh(hidden_dim, 0.0);
//             for (int j = 0; j < hidden_dim; ++j) {
//                 double sum = 0.0;
//                 for (int i = 0; i < output_dim; ++i)
//                     sum += Why[i][j] * dy[i];
//                 dh[j] = sum + dh_next[j];
//             }

//             vector<double> da(hidden_dim, 0.0);
//             for (int j = 0; j < hidden_dim; ++j) {
//                 da[j] = dh[j] * dtanh_from_h(hs[t+1][j]);
//                 dbh[j] += da[j];
//                 for (int i = 0; i < input_dim; ++i)
//                     dWxh[j][i] += da[j] * input_seq[t][i];
//                 for (int i = 0; i < hidden_dim; ++i)
//                     dWhh[j][i] += da[j] * hs[t][i];
//             }

//             // dh_next = Whh^T * da
//             fill(dh_next.begin(), dh_next.end(), 0.0);
//             for (int i = 0; i < hidden_dim; ++i) {
//                 double sum = 0.0;
//                 for (int j = 0; j < hidden_dim; ++j)
//                     sum += Whh[j][i] * da[j];
//                 dh_next[i] = sum;
//             }
//         }

//         // gradient step
//         for (int i = 0; i < hidden_dim; ++i) {
//             bh[i] -= LR * dbh[i];
//             for (int j = 0; j < input_dim; ++j)
//                 Wxh[i][j] -= LR * dWxh[i][j];
//             for (int j = 0; j < hidden_dim; ++j)
//                 Whh[i][j] -= LR * dWhh[i][j];
//         }
//         for (int i = 0; i < output_dim; ++i) {
//             by[i] -= LR * dby[i];
//             for (int j = 0; j < hidden_dim; ++j)
//                 Why[i][j] -= LR * dWhy[i][j];
//         }

//         return total_loss;
//     }

//     // Predict sequence (greedy argmax per step)
//     string predict(const vector<vector<double>>& input_seq, const string& vocab) {
//         vector<double> h(hidden_dim, 0.0);
//         string result;
//         for (const auto& x : input_seq) {
//             vector<double> h_new(hidden_dim, 0.0);
//             for (int i = 0; i < hidden_dim; ++i) {
//                 double a = bh[i];
//                 for (int j = 0; j < input_dim; ++j) a += Wxh[i][j] * x[j];
//                 for (int j = 0; j < hidden_dim; ++j) a += Whh[i][j] * h[j];
//                 h_new[i] = tanh_(a);
//             }
//             h = h_new;

//             vector<double> logits(output_dim, 0.0);
//             for (int i = 0; i < output_dim; ++i) {
//                 double s = by[i];
//                 for (int j = 0; j < hidden_dim; ++j)
//                     s += Why[i][j] * h[j];
//                 logits[i] = s;
//             }
//             vector<double> p = softmax(logits);
//             int idx = argmax(p);
//             result.push_back(vocab[idx]);
//         }
//         return result;
//     }

//     // save and load model for fast inference
//     void save_model(const string& filename) {
//         ofstream ofs(filename, ios::binary);
//         ofs.write(reinterpret_cast<const char*>(&input_dim), sizeof(input_dim));
//         ofs.write(reinterpret_cast<const char*>(&hidden_dim), sizeof(hidden_dim));
//         ofs.write(reinterpret_cast<const char*>(&output_dim), sizeof(output_dim));
//         for (const auto& row : Wxh)
//             ofs.write(reinterpret_cast<const char*>(row.data()), row.size() * sizeof(double));
//         for (const auto& row : Whh)
//             ofs.write(reinterpret_cast<const char*>(row.data()), row.size() * sizeof(double));
//         for (const auto& row : Why)
//             ofs.write(reinterpret_cast<const char*>(row.data()), row.size() * sizeof(double));
//         ofs.write(reinterpret_cast<const char*>(bh.data()), bh.size() * sizeof(double));
//         ofs.write(reinterpret_cast<const char*>(by.data()), by.size() * sizeof(double));
//     }

//     void load_model(const string& filename) {
//         ifstream ifs(filename, ios::binary);
//         if (!ifs) {
//             cerr << "[Error] Could not open model file: " << filename << "\n";
//             return;
//         }
//         ifs.read(reinterpret_cast<char*>(&input_dim), sizeof(input_dim));
//         ifs.read(reinterpret_cast<char*>(&hidden_dim), sizeof(hidden_dim));
//         ifs.read(reinterpret_cast<char*>(&output_dim), sizeof(output_dim));
//         Wxh.assign(hidden_dim, vector<double>(input_dim));
//         Whh.assign(hidden_dim, vector<double>(hidden_dim));
//         Why.assign(output_dim, vector<double>(hidden_dim));
//         bh.assign(hidden_dim, 0.0);
//         by.assign(output_dim, 0.0);
//         for (auto& row : Wxh)
//             ifs.read(reinterpret_cast<char*>(row.data()), row.size() * sizeof(double));
//         for (auto& row : Whh)
//             ifs.read(reinterpret_cast<char*>(row.data()), row.size() * sizeof(double));
//         for (auto& row : Why)
//             ifs.read(reinterpret_cast<char*>(row.data()), row.size() * sizeof(double));
//         ifs.read(reinterpret_cast<char*>(bh.data()), bh.size() * sizeof(double));
//         ifs.read(reinterpret_cast<char*>(by.data()), by.size() * sizeof(double));
//     }

//     // estimated parameter memory (in bytes)
//     size_t param_bytes() const {
//         size_t sz = 0;
//         sz += (size_t)input_dim * hidden_dim * sizeof(double);   // Wxh
//         sz += (size_t)hidden_dim * hidden_dim * sizeof(double);  // Whh
//         sz += (size_t)output_dim * hidden_dim * sizeof(double);  // Why
//         sz += (size_t)hidden_dim * sizeof(double);               // bh
//         sz += (size_t)output_dim * sizeof(double);               // by
//         return sz;
//     }
// };

// // ----------------------
// // Build input seq from obf + (seed,counter)
// // ----------------------
// vector<vector<double>> build_input_seq(const string& obf,
//                                        const string& vocab,
//                                        int seed,
//                                        int counter) {
//     int V = (int)vocab.size();
//     int input_dim = V + 2;
//     vector<vector<double>> x_seq;
//     x_seq.reserve(obf.size());

//     double s_norm = seed / 16.0;    // arbitrary normalization for demo
//     double c_norm = counter / 64.0;

//     for (char c : obf) {
//         vector<double> x(input_dim, 0.0);
//         int idx = (int)vocab.find(c);
//         if (idx < 0) idx = 0;
//         x[idx] = 1.0;
//         x[V]   = s_norm;
//         x[V+1] = c_norm;
//         x_seq.push_back(x);
//     }
//     return x_seq;
// }

// // ----------------------
// // PIM chain monitor WITH seed window control
// // ----------------------
// void pim_chain_monitor(RNN& net,
//                        const string& master,
//                        const string& vocab,
//                        const Obfuscator& obf,
//                        int seed,
//                        pair<int,int> allowed_seed_range,
//                        int start_counter,
//                        int window_size,
//                        int length,
//                        double err_threshold) {
//     int L = (int)master.size();

//     // Seed window control: if seed outside allowed range -> chain break
//     bool seed_ok = (seed >= allowed_seed_range.first &&
//                     seed <= allowed_seed_range.second);

//     cout << "\n=== PIM CHAIN MONITOR ===\n";
//     cout << "seed=" << seed
//          << " allowed_seed_range=[" << allowed_seed_range.first
//          << "," << allowed_seed_range.second << "]\n";

//     if (!seed_ok) {
//         cout << "[CHAIN_BREAK_SEED] seed out of allowed range. "
//              << "No valid PIM chain for this seed.\n";
//         return;
//     }

//     int window_start = start_counter;
//     int window_end   = start_counter + window_size - 1;

//     cout << "counter_window=[" << window_start << "," << window_end << "] "
//          << "length=" << length << " err_threshold=" << err_threshold << "\n";

//     for (int i = 0; i < length; ++i) {
//         int counter = start_counter + i;
//         string obf_str = obf.obfuscate(master, seed, counter);
//         auto x_seq = build_input_seq(obf_str, vocab, seed, counter);
//         string recon = net.predict(x_seq, vocab);

//         int mismatches = 0;
//         for (int t = 0; t < L; ++t) {
//             if (recon[t] != master[t]) mismatches++;
//         }
//         double err_rate = (double)mismatches / (double)L;
//         bool inside_window = (counter >= window_start && counter <= window_end);
//         bool ok = inside_window && (err_rate <= err_threshold);

//         string status;
//         if (!inside_window) status = "OUT_OF_WINDOW_COUNTER";
//         else if (!ok)       status = "ANOMALY";
//         else                status = "OK";

//         cout << " ctr=" << setw(3) << counter
//              << " err_rate=" << fixed << setprecision(3) << err_rate
//              << " inside_window=" << (inside_window ? "true " : "false")
//              << " status=" << status << "\n";
//     }
// }

// // ----------------------
// // MAIN: PIM demo
// // ----------------------
// int main() {
//     ios::sync_with_stdio(false);
//     cin.tie(nullptr);

//     // Vocabulary and random master secret
//     string vocab = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz1234567890!@#$%^&*()-_=+[]{}|;:',.<>?/";
//     int vocab_size = (int)vocab.size();
//     int secret_size = 32;

//     srand((unsigned)time(nullptr));
//     string master;
//     master.reserve(secret_size);
//     for (int i = 0; i < secret_size; ++i)
//         master.push_back(vocab[rand() % vocab_size]);

//     cout << "[Master secret]: " << master << "\n";

//     // Obfuscator
//     Obfuscator obf(vocab);

//     // RNN: input_dim = vocab + 2 (seed, counter), output_dim = vocab
//     int hidden_dim = 32;
//     int input_dim = vocab_size + 2;
//     int output_dim = vocab_size;
//     RNN net(input_dim, hidden_dim, output_dim);

//     // Print model size
//     size_t bytes = net.param_bytes();
//     cout << "[Model params]: " << bytes << " bytes ("
//          << fixed << setprecision(2) << (bytes / 1024.0) << " KB)\n";

//     // Build target indices once (master chars)
//     vector<int> target_indices(secret_size);
//     for (int i = 0; i < secret_size; ++i) {
//         int idx = (int)vocab.find(master[i]);
//         if (idx < 0) idx = 0;
//         target_indices[i] = idx;
//     }

//     // Training hyperparameters
//     int epochs = 250;
//     pair<int,int> seed_range    = {1, 3};  // <-- TRAINED seed window
//     pair<int,int> counter_range = {1, 8};  // <-- TRAINED counters

//     cout << "\n=== TRAINING (obf -> master) ===\n";

//     clock_t train_start = clock();
//     double sum_epoch_ms = 0.0;

//     for (int epoch = 0; epoch <= epochs; ++epoch) {
//         clock_t epoch_start = clock();

//         double total_loss = 0.0;
//         int num_pairs = 0;

//         for (int s = seed_range.first; s <= seed_range.second; ++s) {
//             for (int c = counter_range.first; c <= counter_range.second; ++c) {
//                 string obf_str = obf.obfuscate(master, s, c);
//                 auto x_seq = build_input_seq(obf_str, vocab, s, c);
//                 double loss = net.forward_backward(x_seq, target_indices);
//                 total_loss += loss;
//                 num_pairs++;
//             }
//         }

//         double avg_loss = total_loss / (double)num_pairs;

//         clock_t epoch_end = clock();
//         double epoch_ms = double(epoch_end - epoch_start) / CLOCKS_PER_SEC * 1000.0;
//         sum_epoch_ms += epoch_ms;

//         if (epoch % 10 == 0) {
//             cout << "[Epoch " << setw(3) << epoch << "] avg_loss=" << avg_loss
//                  << " | epoch_time=" << fixed << setprecision(3) << epoch_ms << " ms\n";
//         }
//     }

//     clock_t train_end = clock();
//     double train_ms = double(train_end - train_start) / CLOCKS_PER_SEC * 1000.0;
//     double avg_epoch_ms = sum_epoch_ms / double(epochs + 1);

//     cout << "[Time train - total]: " << fixed << setprecision(3) << train_ms << " ms\n";
//     cout << "[Time train - avg/epoch]: " << fixed << setprecision(3) << avg_epoch_ms << " ms\n";

//     // Test reconstruction for one (seed, counter) inside training range
//     int test_seed = 2;
//     int test_counter = 7;
//     string obf_test = obf.obfuscate(master, test_seed, test_counter);
//     auto x_test = build_input_seq(obf_test, vocab, test_seed, test_counter);
//     string recon = net.predict(x_test, vocab);

//     cout << "\n=== ROUNDTRIP TEST ===\n";
//     cout << "Master       : " << master << "\n";
//     cout << "Obfuscated   : " << obf_test << "\n";
//     cout << "Reconstructed: " << recon << "\n";
//     cout << "Equal?       : " << ((recon == master) ? "true" : "false") << "\n";

//     // Save model
//     net.save_model("model.bin");

//     // Load model into fresh instance and time fast inference
//     clock_t load_infer_start = clock();
//     RNN net_loaded(1,1,1);  // dummy, overwritten by load
//     net_loaded.load_model("model.bin");
//     auto x_test2 = build_input_seq(obf_test, vocab, test_seed, test_counter);
//     string recon2 = net_loaded.predict(x_test2, vocab);
//     clock_t load_infer_end = clock();
//     double load_infer_ms = double(load_infer_end - load_infer_start) / CLOCKS_PER_SEC * 1000.0;

//     cout << "\n=== FAST INFERENCE (load + predict) ===\n";
//     cout << "Reconstructed (loaded): " << recon2 << "\n";
//     cout << "Equal?                 : " << ((recon2 == master) ? "true" : "false") << "\n";
//     cout << "Time (load+infer)      : " << fixed << setprecision(3) << load_infer_ms << " ms\n";

//     // PIM behavior analysis over drifting window of counters
//     // NOTE: we pass the TRAINED seed_range so if we call with seed outside this range,
//     //       pim_chain_monitor will raise CHAIN_BREAK_SEED.
//     pim_chain_monitor(net_loaded,
//                       master,
//                       vocab,
//                       obf,
//                       /*seed*/ test_seed,
//                       /*allowed_seed_range*/ seed_range,
//                       /*start_counter*/ 1,
//                       /*window_size*/ 8,   // legal counters: 1..8
//                       /*length*/ 16,
//                       /*err_threshold*/ 0.25);

//     // Example: this will trigger CHAIN_BREAK_SEED, because seed=10 is outside [1,3]
//     cout << "\n=== If PIM CHAIN with INVALID SEED (it should break) ===\n";
//     pim_chain_monitor(net_loaded,
//                       master,
//                       vocab,
//                       obf,
//                       /*seed*/ 2,
//                       /*allowed_seed_range*/ seed_range,
//                       /*start_counter*/ 1,
//                       /*window_size*/ 8,
//                       /*length*/ 8,
//                       /*err_threshold*/ 0.25);

//     return 0;
// }
// --------------------------------------------------------
// WARL0K PIM demo with seed+counter window control
// Core: RNN-style autoencoder for deterministic reconstruction
// Task: obfuscated_secret -> master_secret with PIM-style behavior monitoring
// warlok_pim_gru_attn_pn_infer.cpp
// ------------------------------------------------------------
// WARL0K PIM — NumPy reference C++ inference port (GRU + Attn)
// - Loads a trained nano-model from a compact binary file
// - Runs inference in RAM and prints: id/window predictions + p_valid
// - Measures: model load time + single inference time (ms)
// - Prints: stored model size (bytes)
//
// This file is inference-only (no training).
//
// Expected companion: a Python exporter (included below as a comment)
// that writes model weights into the exact binary layout this program reads.
//
// Build:
//   g++ -O3 -std=c++17 warlok_pim_gru_attn_pn_infer.cpp -o warlok_pim_infer
//
// Run:
//   ./warlok_pim_infer model.bin
//
// ------------------------------------------------------------

// pim_warlok.cpp
// Single-file C++ WARL0K PIM demo: GRU + Attention, 2-phase training, save/load, timing & sizing.
// No external deps (standard library only).
//
// Based on the Python prototype structure and logic. :contentReference[oaicite:1]{index=1}

// #include <iostream>
// #include <vector>
// #include <string>
// #include <cmath>
// #include <cstdint>
// #include <fstream>
// #include <chrono>
// #include <algorithm>
// #include <numeric>
// #include <limits>
// #include <random>

// using std::cout;
// using std::endl;

// // ============================================================
// // Config (match the spirit of the Python prototype)
// // ============================================================
// static constexpr int VOCAB_SIZE       = 16;
// static constexpr int MS_DIM           = 8;
// static constexpr int SEQ_LEN          = 20;

// static constexpr int N_IDENTITIES     = 6;
// static constexpr int N_WINDOWS_PER_ID = 48;

// static constexpr int HIDDEN_DIM       = 64;
// static constexpr int ATTN_DIM         = 32;
// static constexpr int MS_HID           = 32;

// static constexpr int BATCH_SIZE       = 32;

// // You can tune these
// static constexpr int EPOCHS_PHASE1    = 120;  // (Python used 200)
// static constexpr int EPOCHS_PHASE2    = 250;  // (Python used 500)

// static constexpr float LR_PHASE1      = 0.006f;
// // static constexpr float LR_PHASE2_BASE = 0.03f;
// static constexpr float LR_PHASE2_BASE = 0.008f;


// static constexpr float CLIP_NORM      = 5.0f;
// static constexpr float WEIGHT_DECAY   = 1e-4f;

// // Phase1 losses
// static constexpr float LAMBDA_MS      = 1.0f;
// static constexpr float LAMBDA_TOK     = 0.10f;
// static constexpr float TOK_STOP_EPS   = 0.25f;
// static constexpr int   TOK_WARMUP_EPOCHS = 60;

// // Phase2 losses
// static constexpr float LAMBDA_ID      = 1.0f;
// static constexpr float LAMBDA_W       = 1.0f;
// static constexpr float LAMBDA_BCE     = 1.0f;
// static constexpr float POS_WEIGHT     = 10.0f;

// // Strong accept thresholds (demo gating)
// static constexpr float THRESH_P_VALID = 0.80f;
// static constexpr float PID_MIN        = 0.70f;
// static constexpr float PW_MIN         = 0.70f;

// // Window pilot (PN watermark)
// static constexpr float PILOT_AMP      = 0.55f;

// // Backbone input dim: one-hot vocab + meas + time
// static constexpr int INPUT_DIM        = VOCAB_SIZE + 2;

// // ============================================================
// // Small utilities
// // ============================================================
// static inline float sigmoid(float x) {
//     // stable-ish
//     if (x >= 0) {
//         float z = std::exp(-x);
//         return 1.0f / (1.0f + z);
//     } else {
//         float z = std::exp(x);
//         return z / (1.0f + z);
//     }
// }

// static inline float fast_tanh(float x) {
//     return std::tanh(x);
// }

// static inline float clampf(float x, float lo, float hi) {
//     return std::max(lo, std::min(hi, x));
// }

// // deterministic xorshift32
// struct XorShift32 {
//     uint32_t s;
//     explicit XorShift32(uint32_t seed=0x12345678u) : s(seed ? seed : 0x12345678u) {}
//     uint32_t next_u32() {
//         uint32_t x = s;
//         x ^= x << 13;
//         x ^= x >> 17;
//         x ^= x << 5;
//         s = x;
//         return x;
//     }
//     float next_f01() { // [0,1)
//         return (next_u32() >> 8) * (1.0f / 16777216.0f);
//     }
//     int next_int(int lo, int hi) { // inclusive lo, exclusive hi
//         uint32_t r = next_u32();
//         return lo + int(r % uint32_t(hi - lo));
//     }
//     // approx normal via Box-Muller
//     float next_norm() {
//         float u1 = std::max(1e-7f, next_f01());
//         float u2 = next_f01();
//         float r = std::sqrt(-2.0f * std::log(u1));
//         float t = 2.0f * 3.1415926535f * u2;
//         return r * std::cos(t);
//     }
// };

// static inline double now_seconds() {
//     using clk = std::chrono::high_resolution_clock;
//     auto t = clk::now().time_since_epoch();
//     return std::chrono::duration<double>(t).count();
// }

// // ============================================================
// // Lightweight tensor wrappers (flat vectors + index helpers)
// // ============================================================
// struct Mat {
//     int R=0, C=0;
//     std::vector<float> a; // row-major
//     Mat() {}
//     Mat(int r,int c,float v=0):R(r),C(c),a(size_t(r)*c,v) {}
//     float& operator()(int r,int c){ return a[size_t(r)*C + c]; }
//     const float& operator()(int r,int c) const { return a[size_t(r)*C + c]; }
// };

// struct Vec {
//     int N=0;
//     std::vector<float> a;
//     Vec() {}
//     Vec(int n,float v=0):N(n),a(size_t(n),v) {}
//     float& operator()(int i){ return a[size_t(i)]; }
//     const float& operator()(int i) const { return a[size_t(i)]; }
// };

// // static inline void zeros(Mat& m){ std::fill(m.a.begin(), m.a.end(), 0.0f); }
// // static inline void zeros(Vec& v){ std::fill(v.a.begin(), v.a.end(), 0.0f); }

// // L2 norm between two vectors
// static float l2_vec(const std::vector<float>& x, const std::vector<float>& y) {
//     float s=0;
//     for(size_t i=0;i<x.size();++i){
//         float d = x[i]-y[i];
//         s += d*d;
//     }
//     return std::sqrt(s);
// }

// // ============================================================
// // Dataset generation (MS_all, A_base, window_delta, PN pilot)
// // ============================================================
// static Mat MS_all;    // [N_IDENTITIES, MS_DIM]
// static Mat A_base;    // [SEQ_LEN, MS_DIM]

// static std::vector<float> window_delta_vec(int window_global_id, int t) {
//     uint32_t seed = uint32_t((window_global_id * 10007 + t * 97) & 0xFFFFFFFF);
//     XorShift32 rng(seed ? seed : 0xA5A5A5A5u);
//     std::vector<float> d(MS_DIM);
//     for(int i=0;i<MS_DIM;i++){
//         d[i] = 0.25f * rng.next_norm();
//     }
//     return d;
// }

// static std::vector<float> window_pilot_vec(int window_global_id) {
//     uint32_t seed = uint32_t((window_global_id * 9176 + 11) & 0xFFFFFFFF);
//     XorShift32 rng(seed ? seed : 0xBEEFBEEFu);
//     std::vector<float> pilot(SEQ_LEN, 0.0f);
//     float mean=0.0f;
//     for(int t=0;t<SEQ_LEN;t++){
//         int bit = rng.next_int(0,2); // 0/1
//         float chip = (bit==0) ? -1.0f : 1.0f;
//         pilot[t] = PILOT_AMP * chip;
//         mean += pilot[t];
//     }
//     mean /= float(SEQ_LEN);
//     for(int t=0;t<SEQ_LEN;t++) pilot[t] -= mean;
//     return pilot;
// }

// // Generate OS chain -> tokens + normalized measurements m[t]
// static void generate_os_chain(
//     const std::vector<float>& ms_vec,
//     int window_global_id,
//     std::vector<int>& tokens_out,
//     std::vector<float>& meas_out
// ) {
//     std::vector<float> zs(SEQ_LEN, 0.0f);

//     // core projection with per-window delta
//     for(int t=0;t<SEQ_LEN;t++){
//         auto d = window_delta_vec(window_global_id, t);
//         float dot=0.0f;
//         for(int k=0;k<MS_DIM;k++){
//             float a = A_base(t,k) + d[k];
//             dot += a * ms_vec[k];
//         }
//         zs[t] = dot;
//     }

//     // add PN pilot watermark
//     auto pilot = window_pilot_vec(window_global_id);
//     for(int t=0;t<SEQ_LEN;t++) zs[t] += pilot[t];

//     // small noise (seeded)
//     int ms_sum = 0;
//     for(int k=0;k<MS_DIM;k++) ms_sum += int(ms_vec[k]*1000.0f);
//     uint32_t nseed = uint32_t((window_global_id * 1337 + ms_sum) & 0xFFFFFFFF);
//     XorShift32 rng(nseed ? nseed : 0xCAFE1234u);
//     for(int t=0;t<SEQ_LEN;t++){
//         zs[t] += 0.02f * rng.next_norm();
//     }

//     // normalize zs -> m
//     float mean=0.0f;
//     for(float v: zs) mean += v;
//     mean /= float(SEQ_LEN);

//     float var=0.0f;
//     for(float v: zs){
//         float d = v - mean;
//         var += d*d;
//     }
//     var /= float(SEQ_LEN);
//     float st = std::sqrt(var) + 1e-6f;

//     meas_out.assign(SEQ_LEN, 0.0f);
//     for(int t=0;t<SEQ_LEN;t++){
//         meas_out[t] = (zs[t] - mean) / st;
//     }

//     // quantize
//     tokens_out.assign(SEQ_LEN, 0);
//     for(int t=0;t<SEQ_LEN;t++){
//         float scaled = clampf((meas_out[t] + 3.0f) / 6.0f, 0.0f, 0.999999f);
//         int tok = int(scaled * float(VOCAB_SIZE));
//         if(tok < 0) tok = 0;
//         if(tok >= VOCAB_SIZE) tok = VOCAB_SIZE-1;
//         tokens_out[t] = tok;
//     }
// }

// static void build_X_backbone(
//     const std::vector<int>& tokens,
//     const std::vector<float>& meas,
//     Mat& X_out  // [SEQ_LEN, INPUT_DIM]
// ) {
//     X_out = Mat(SEQ_LEN, INPUT_DIM, 0.0f);
//     for(int t=0;t<SEQ_LEN;t++){
//         int tok = tokens[t];
//         X_out(t, tok) = 1.0f;
//         X_out(t, VOCAB_SIZE) = meas[t];
//         X_out(t, VOCAB_SIZE + 1) = (SEQ_LEN <= 1) ? 0.0f : (float(t) / float(SEQ_LEN-1));
//     }
// }

// struct Dataset {
//     // X: [N, T, D] -> stored as flat contiguous [N][T][D]
//     int N=0;
//     std::vector<float> X;     // size N*SEQ_LEN*INPUT_DIM
//     std::vector<float> M;     // mask size N*SEQ_LEN
//     std::vector<int>   TOK;   // tokens size N*SEQ_LEN
//     std::vector<float> Y_MS;  // size N*MS_DIM
//     std::vector<float> Y_CLS; // size N
//     std::vector<int> TRUE_ID, TRUE_W, CLAIM_ID, EXPECT_W;
// };

// static inline size_t idx3(int n,int t,int d,int T,int D){
//     return (size_t(n)*T + t)*D + d;
// }
// static inline size_t idx2(int n,int t,int T){
//     return size_t(n)*T + t;
// }

// static Dataset build_dataset() {
//     Dataset ds;

//     // 1 positive + 4 negatives per (id, w) in Python prototype => 5 samples
//     const int samples_per_pair = 5;
//     ds.N = N_IDENTITIES * N_WINDOWS_PER_ID * samples_per_pair;

//     ds.X.assign(size_t(ds.N)*SEQ_LEN*INPUT_DIM, 0.0f);
//     ds.M.assign(size_t(ds.N)*SEQ_LEN, 0.0f);
//     ds.TOK.assign(size_t(ds.N)*SEQ_LEN, 0);
//     ds.Y_MS.assign(size_t(ds.N)*MS_DIM, 0.0f);
//     ds.Y_CLS.assign(size_t(ds.N), 0.0f);

//     ds.TRUE_ID.assign(ds.N, 0);
//     ds.TRUE_W.assign(ds.N, 0);
//     ds.CLAIM_ID.assign(ds.N, 0);
//     ds.EXPECT_W.assign(ds.N, 0);

//     XorShift32 rng(0xBADC0DEu);

//     int n=0;
//     for(int id_true=0; id_true<N_IDENTITIES; id_true++){
//         std::vector<float> ms_true(MS_DIM);
//         for(int k=0;k<MS_DIM;k++) ms_true[k] = MS_all(id_true,k);

//         for(int w_true=0; w_true<N_WINDOWS_PER_ID; w_true++){
//             int g_true = id_true * N_WINDOWS_PER_ID + w_true;

//             std::vector<int> toks;
//             std::vector<float> meas;
//             generate_os_chain(ms_true, g_true, toks, meas);

//             auto push_sample = [&](const std::vector<int>& toks_in,
//                                    const std::vector<float>& meas_in,
//                                    float ycls,
//                                    int true_id, int true_w,
//                                    int claim_id, int expect_w,
//                                    const std::vector<float>& yms){
//                 // X
//                 Mat X;
//                 build_X_backbone(toks_in, meas_in, X);
//                 for(int t=0;t<SEQ_LEN;t++){
//                     ds.M[idx2(n,t,SEQ_LEN)] = 1.0f;
//                     ds.TOK[idx2(n,t,SEQ_LEN)] = toks_in[t];
//                     for(int d=0; d<INPUT_DIM; d++){
//                         ds.X[idx3(n,t,d,SEQ_LEN,INPUT_DIM)] = X(t,d);
//                     }
//                 }
//                 // Y
//                 for(int k=0;k<MS_DIM;k++){
//                     ds.Y_MS[size_t(n)*MS_DIM + k] = yms[k];
//                 }
//                 ds.Y_CLS[size_t(n)] = ycls;

//                 ds.TRUE_ID[n] = true_id;
//                 ds.TRUE_W[n] = true_w;
//                 ds.CLAIM_ID[n] = claim_id;
//                 ds.EXPECT_W[n] = expect_w;

//                 n++;
//             };

//             // POS legit
//             push_sample(toks, meas, 1.0f, id_true, w_true, id_true, w_true, ms_true);

//             // NEG shuffled
//             {
//                 std::vector<int> idxs(SEQ_LEN);
//                 std::iota(idxs.begin(), idxs.end(), 0);
//                 std::shuffle(idxs.begin(), idxs.end(),
//                     std::mt19937(0x1234 + g_true)); // deterministic
//                 std::vector<int> toks2(SEQ_LEN);
//                 std::vector<float> meas2(SEQ_LEN);
//                 for(int t=0;t<SEQ_LEN;t++){ toks2[t]=toks[idxs[t]]; meas2[t]=meas[idxs[t]]; }
//                 push_sample(toks2, meas2, 0.0f, id_true, w_true, id_true, w_true, ms_true);
//             }

//             // NEG truncated (half filled, rest zero-mask)
//             {
//                 int Ltr = SEQ_LEN/2;
//                 std::vector<int> toks2(SEQ_LEN, 0);
//                 std::vector<float> meas2(SEQ_LEN, 0.0f);
//                 for(int t=0;t<Ltr;t++){ toks2[t]=toks[t]; meas2[t]=meas[t]; }

//                 // Build X with proper mask (first Ltr are valid)
//                 Mat X;
//                 build_X_backbone(toks2, meas2, X);
//                 // write sample manually with mask
//                 int nn = n;
//                 for(int t=0;t<SEQ_LEN;t++){
//                     ds.M[idx2(nn,t,SEQ_LEN)] = (t < Ltr) ? 1.0f : 0.0f;
//                     ds.TOK[idx2(nn,t,SEQ_LEN)] = toks2[t];
//                     for(int d=0; d<INPUT_DIM; d++){
//                         ds.X[idx3(nn,t,d,SEQ_LEN,INPUT_DIM)] = X(t,d);
//                     }
//                 }
//                 for(int k=0;k<MS_DIM;k++) ds.Y_MS[size_t(nn)*MS_DIM + k] = ms_true[k];
//                 ds.Y_CLS[size_t(nn)] = 0.0f;
//                 ds.TRUE_ID[nn]=id_true; ds.TRUE_W[nn]=w_true;
//                 ds.CLAIM_ID[nn]=id_true; ds.EXPECT_W[nn]=w_true;
//                 n++;
//             }

//             // NEG wrong-window chain (claim expects w_true)
//             {
//                 int wrong_w = (w_true + 7) % N_WINDOWS_PER_ID;
//                 int g_wrong = id_true * N_WINDOWS_PER_ID + wrong_w;
//                 std::vector<int> toks_w;
//                 std::vector<float> meas_w;
//                 generate_os_chain(ms_true, g_wrong, toks_w, meas_w);
//                 push_sample(toks_w, meas_w, 0.0f, id_true, wrong_w, id_true, w_true, ms_true);
//             }

//             // NEG wrong identity chain (claim id_true)
//             {
//                 int other_id = (id_true + rng.next_int(1, N_IDENTITIES)) % N_IDENTITIES;
//                 int other_w  = rng.next_int(0, N_WINDOWS_PER_ID);
//                 int g_other  = other_id * N_WINDOWS_PER_ID + other_w;
//                 std::vector<float> ms_other(MS_DIM);
//                 for(int k=0;k<MS_DIM;k++) ms_other[k] = MS_all(other_id,k);
//                 std::vector<int> toks_o;
//                 std::vector<float> meas_o;
//                 generate_os_chain(ms_other, g_other, toks_o, meas_o);
//                 // y_ms stays ms_true (as in prototype), so mismatch should fail
//                 push_sample(toks_o, meas_o, 0.0f, other_id, other_w, id_true, w_true, ms_true);
//             }
//         }
//     }

//     return ds;
// }

// // ============================================================
// // Model params
// // ============================================================
// struct Params {
//     // GRU
//     Mat W_z, U_z; Vec b_z;
//     Mat W_r, U_r; Vec b_r;
//     Mat W_h, U_h; Vec b_h;

//     // Attn
//     Mat W_att; Vec v_att;

//     // MS head
//     Mat W_ms1; Vec b_ms1;
//     Mat W_ms2; Vec b_ms2;

//     // Token scaffold
//     Mat W_tok; Vec b_tok;

//     // Heads
//     Mat W_id; Vec b_id;
//     Mat W_w;  Vec b_w;

//     Mat W_beh; Vec b_beh; // W_beh: [1, HIDDEN_DIM+4]

//     // for sizing
//     size_t param_count() const {
//         auto count_mat = [](const Mat& m){ return size_t(m.R)*m.C; };
//         auto count_vec = [](const Vec& v){ return size_t(v.N); };

//         size_t c=0;
//         c += count_mat(W_z)+count_mat(U_z)+count_vec(b_z);
//         c += count_mat(W_r)+count_mat(U_r)+count_vec(b_r);
//         c += count_mat(W_h)+count_mat(U_h)+count_vec(b_h);
//         c += count_mat(W_att)+count_vec(v_att);
//         c += count_mat(W_ms1)+count_vec(b_ms1);
//         c += count_mat(W_ms2)+count_vec(b_ms2);
//         c += count_mat(W_tok)+count_vec(b_tok);
//         c += count_mat(W_id)+count_vec(b_id);
//         c += count_mat(W_w)+count_vec(b_w);
//         c += count_mat(W_beh)+count_vec(b_beh);
//         return c;
//     }
//     size_t bytes() const { return param_count() * sizeof(float); }
// };

// static float randn(XorShift32& rng) { return rng.next_norm(); }

// static Params init_model(int input_dim, uint32_t seed=0xC0FFEEu) {
//     Params p;
//     XorShift32 rng(seed);

//     auto init_mat = [&](int r,int c,float s){
//         Mat m(r,c,0.0f);
//         for(auto& x: m.a) x = s * randn(rng);
//         return m;
//     };
//     auto init_vec0 = [&](int n){
//         return Vec(n, 0.0f);
//     };

//     float s = 0.08f;

//     p.W_z = init_mat(HIDDEN_DIM, input_dim, s);
//     p.U_z = init_mat(HIDDEN_DIM, HIDDEN_DIM, s);
//     p.b_z = init_vec0(HIDDEN_DIM);

//     p.W_r = init_mat(HIDDEN_DIM, input_dim, s);
//     p.U_r = init_mat(HIDDEN_DIM, HIDDEN_DIM, s);
//     p.b_r = init_vec0(HIDDEN_DIM);

//     p.W_h = init_mat(HIDDEN_DIM, input_dim, s);
//     p.U_h = init_mat(HIDDEN_DIM, HIDDEN_DIM, s);
//     p.b_h = init_vec0(HIDDEN_DIM);

//     p.W_att = init_mat(ATTN_DIM, HIDDEN_DIM, s);
//     p.v_att = Vec(ATTN_DIM, 0.0f);
//     for(auto& x: p.v_att.a) x = s * randn(rng);

//     p.W_ms1 = init_mat(MS_HID, HIDDEN_DIM, s);
//     p.b_ms1 = init_vec0(MS_HID);
//     p.W_ms2 = init_mat(MS_DIM, MS_HID, s);
//     p.b_ms2 = init_vec0(MS_DIM);

//     p.W_tok = init_mat(VOCAB_SIZE, HIDDEN_DIM, s);
//     p.b_tok = init_vec0(VOCAB_SIZE);

//     p.W_id = init_mat(N_IDENTITIES, HIDDEN_DIM, s);
//     p.b_id = init_vec0(N_IDENTITIES);

//     p.W_w  = init_mat(N_WINDOWS_PER_ID, 3*HIDDEN_DIM, s);
//     p.b_w  = init_vec0(N_WINDOWS_PER_ID);

//     p.W_beh = init_mat(1, HIDDEN_DIM+4, s);
//     p.b_beh = init_vec0(1);

//     return p;
// }

// // ============================================================
// // Adam optimizer (per-parameter buffers)
// // ============================================================
// struct AdamState {
//     // mirrors Params shapes
//     Params m, v;
//     int t=0;
//     float lr=0.001f;
//     float b1=0.9f, b2=0.999f, eps=1e-8f;

//     static Params zeros_like(const Params& p) {
//         Params z;
//         auto zmat = [](const Mat& m){ return Mat(m.R,m.C,0.0f); };
//         auto zvec = [](const Vec& v){ return Vec(v.N,0.0f); };

//         z.W_z=zmat(p.W_z); z.U_z=zmat(p.U_z); z.b_z=zvec(p.b_z);
//         z.W_r=zmat(p.W_r); z.U_r=zmat(p.U_r); z.b_r=zvec(p.b_r);
//         z.W_h=zmat(p.W_h); z.U_h=zmat(p.U_h); z.b_h=zvec(p.b_h);
//         z.W_att=zmat(p.W_att); z.v_att=zvec(p.v_att);
//         z.W_ms1=zmat(p.W_ms1); z.b_ms1=zvec(p.b_ms1);
//         z.W_ms2=zmat(p.W_ms2); z.b_ms2=zvec(p.b_ms2);
//         z.W_tok=zmat(p.W_tok); z.b_tok=zvec(p.b_tok);
//         z.W_id=zmat(p.W_id); z.b_id=zvec(p.b_id);
//         z.W_w=zmat(p.W_w); z.b_w=zvec(p.b_w);
//         z.W_beh=zmat(p.W_beh); z.b_beh=zvec(p.b_beh);
//         return z;
//     }

//     explicit AdamState(const Params& p, float lr_) : m(zeros_like(p)), v(zeros_like(p)), lr(lr_) {}

//     static inline void step_mat(Mat& w, const Mat& g, Mat& m, Mat& v,
//                                 int t, float lr, float b1, float b2, float eps, float wd, bool bias) {
//         const float b1t = std::pow(b1, float(t));
//         const float b2t = std::pow(b2, float(t));
//         for(size_t i=0;i<w.a.size();++i){
//             float grad = g.a[i];
//             if(wd > 0.0f && !bias) grad += wd * w.a[i];

//             m.a[i] = b1*m.a[i] + (1-b1)*grad;
//             v.a[i] = b2*v.a[i] + (1-b2)*grad*grad;

//             float mhat = m.a[i] / (1 - b1t);
//             float vhat = v.a[i] / (1 - b2t);
//             w.a[i] -= lr * mhat / (std::sqrt(vhat) + eps);
//         }
//     }

//     static inline void step_vec(Vec& w, const Vec& g, Vec& m, Vec& v,
//                                 int t, float lr, float b1, float b2, float eps, float wd, bool bias) {
//         const float b1t = std::pow(b1, float(t));
//         const float b2t = std::pow(b2, float(t));
//         for(size_t i=0;i<w.a.size();++i){
//             float grad = g.a[i];
//             if(wd > 0.0f && !bias) grad += wd * w.a[i];

//             m.a[i] = b1*m.a[i] + (1-b1)*grad;
//             v.a[i] = b2*v.a[i] + (1-b2)*grad*grad;

//             float mhat = m.a[i] / (1 - b1t);
//             float vhat = v.a[i] / (1 - b2t);
//             w.a[i] -= lr * mhat / (std::sqrt(vhat) + eps);
//         }
//     }

//     // freeze flags for Phase2 backbone
//     struct Freeze {
//         bool W_z=false,U_z=false,b_z=false;
//         bool W_r=false,U_r=false,b_r=false;
//         bool W_h=false,U_h=false,b_h=false;
//         bool W_att=false,v_att=false;
//         bool W_ms1=false,b_ms1=false,W_ms2=false,b_ms2=false;
//         bool W_tok=false,b_tok=false;
//         bool W_id=false,b_id=false;
//         bool W_w=false,b_w=false;
//         bool W_beh=false,b_beh=false;
//     };

//     void step(Params& p, const Params& g, float weight_decay, const Freeze& fr) {
//         t += 1;

//         auto SM = [&](Mat& W, const Mat& dW, Mat& mW, Mat& vW, bool freeze, bool bias=false){
//             if(freeze) return;
//             step_mat(W, dW, mW, vW, t, lr, b1, b2, eps, weight_decay, bias);
//         };
//         auto SV = [&](Vec& b, const Vec& db, Vec& mb, Vec& vb, bool freeze, bool bias=true){
//             if(freeze) return;
//             step_vec(b, db, mb, vb, t, lr, b1, b2, eps, weight_decay, bias);
//         };

//         SM(p.W_z, g.W_z, m.W_z, v.W_z, fr.W_z, false);  SM(p.U_z, g.U_z, m.U_z, v.U_z, fr.U_z, false);  SV(p.b_z, g.b_z, m.b_z, v.b_z, fr.b_z, true);
//         SM(p.W_r, g.W_r, m.W_r, v.W_r, fr.W_r, false);  SM(p.U_r, g.U_r, m.U_r, v.U_r, fr.U_r, false);  SV(p.b_r, g.b_r, m.b_r, v.b_r, fr.b_r, true);
//         SM(p.W_h, g.W_h, m.W_h, v.W_h, fr.W_h, false);  SM(p.U_h, g.U_h, m.U_h, v.U_h, fr.U_h, false);  SV(p.b_h, g.b_h, m.b_h, v.b_h, fr.b_h, true);

//         SM(p.W_att, g.W_att, m.W_att, v.W_att, fr.W_att, false); SV(p.v_att, g.v_att, m.v_att, v.v_att, fr.v_att, true);

//         SM(p.W_ms1, g.W_ms1, m.W_ms1, v.W_ms1, fr.W_ms1, false); SV(p.b_ms1, g.b_ms1, m.b_ms1, v.b_ms1, fr.b_ms1, true);
//         SM(p.W_ms2, g.W_ms2, m.W_ms2, v.W_ms2, fr.W_ms2, false); SV(p.b_ms2, g.b_ms2, m.b_ms2, v.b_ms2, fr.b_ms2, true);

//         SM(p.W_tok, g.W_tok, m.W_tok, v.W_tok, fr.W_tok, false); SV(p.b_tok, g.b_tok, m.b_tok, v.b_tok, fr.b_tok, true);

//         SM(p.W_id, g.W_id, m.W_id, v.W_id, fr.W_id, false); SV(p.b_id, g.b_id, m.b_id, v.b_id, fr.b_id, true);
//         SM(p.W_w,  g.W_w,  m.W_w,  v.W_w,  fr.W_w,  false); SV(p.b_w,  g.b_w,  m.b_w,  v.b_w,  fr.b_w,  true);

//         SM(p.W_beh, g.W_beh, m.W_beh, v.W_beh, fr.W_beh, false); SV(p.b_beh, g.b_beh, m.b_beh, v.b_beh, fr.b_beh, true);
//     }
// };

// // ============================================================
// // Softmax helpers
// // ============================================================
// static void softmax1d(const std::vector<float>& logits, std::vector<float>& probs) {
//     float mx = -1e30f;
//     for(float v: logits) mx = std::max(mx, v);
//     float s=0.0f;
//     probs.resize(logits.size());
//     for(size_t i=0;i<logits.size();++i){
//         float e = std::exp(logits[i]-mx);
//         probs[i]=e; s+=e;
//     }
//     float inv = 1.0f/(s+1e-12f);
//     for(float& p: probs) p*=inv;
// }

// // masked softmax for attention scores: scores[B][T] with mask[B][T]
// static void softmax_masked(const std::vector<float>& scores, const std::vector<float>& mask,
//                            int B, int T, std::vector<float>& alphas) {
//     alphas.assign(size_t(B)*T, 0.0f);
//     for(int b=0;b<B;b++){
//         float mx = -1e30f;
//         for(int t=0;t<T;t++){
//             float m = mask[idx2(b,t,T)];
//             if(m>0.0f) mx = std::max(mx, scores[idx2(b,t,T)]);
//         }
//         float s=0.0f;
//         for(int t=0;t<T;t++){
//             float m = mask[idx2(b,t,T)];
//             if(m<=0.0f) continue;
//             float e = std::exp(scores[idx2(b,t,T)] - mx);
//             alphas[idx2(b,t,T)] = e;
//             s += e;
//         }
//         float inv = 1.0f/(s+1e-12f);
//         for(int t=0;t<T;t++){
//             alphas[idx2(b,t,T)] *= inv;
//         }
//     }
// }

// // ============================================================
// // Forward: GRU (batch) + attention + heads
// // ============================================================
// // We'll store only what we need for backprop (Phase1).
// struct GRUCache {
//     int B=0, T=0, D=0;
//     std::vector<float> X;   // B*T*D
//     std::vector<float> M;   // B*T
//     std::vector<float> H;   // B*T*HIDDEN
//     std::vector<float> Z;   // B*T*HIDDEN
//     std::vector<float> R;   // B*T*HIDDEN
//     std::vector<float> HT;  // B*T*HIDDEN (htilde)
// };

// static void gru_forward_batch(
//     const Params& p,
//     const std::vector<float>& Xb, // B*T*D
//     const std::vector<float>& Mb, // B*T
//     int B, int T, int D,
//     std::vector<float>& H_out, // B*T*HIDDEN
//     GRUCache& cache
// ) {
//     H_out.assign(size_t(B)*T*HIDDEN_DIM, 0.0f);
//     cache = GRUCache();
//     cache.B=B; cache.T=T; cache.D=D;
//     cache.X = Xb;
//     cache.M = Mb;
//     cache.H.assign(size_t(B)*T*HIDDEN_DIM, 0.0f);
//     cache.Z.assign(size_t(B)*T*HIDDEN_DIM, 0.0f);
//     cache.R.assign(size_t(B)*T*HIDDEN_DIM, 0.0f);
//     cache.HT.assign(size_t(B)*T*HIDDEN_DIM, 0.0f);

//     // h_prev[b][h]
//     std::vector<float> h_prev(size_t(B)*HIDDEN_DIM, 0.0f);

//     for(int t=0;t<T;t++){
//         for(int b=0;b<B;b++){
//             float mt = Mb[idx2(b,t,T)];

//             // compute z,r,htil
//             for(int h=0; h<HIDDEN_DIM; h++){
//                 // a_z = W_z*x + U_z*h_prev + b
//                 float az = p.b_z(h);
//                 float ar = p.b_r(h);
//                 float ah = p.b_h(h);

//                 // W_* [H, D] times x [D]
//                 const float* xptr = &Xb[idx3(b,t,0,T,D)];
//                 for(int d=0; d<D; d++){
//                     float x = xptr[d];
//                     az += p.W_z(h,d) * x;
//                     ar += p.W_r(h,d) * x;
//                     ah += p.W_h(h,d) * x;
//                 }
//                 // U_z/U_r uses h_prev
//                 const float* hp = &h_prev[size_t(b)*HIDDEN_DIM];
//                 float uz=0.0f, ur=0.0f, uh=0.0f;
//                 for(int k=0;k<HIDDEN_DIM;k++){
//                     float hv = hp[k];
//                     uz += p.U_z(h,k) * hv;
//                     ur += p.U_r(h,k) * hv;
//                 }

//                 float z = sigmoid(az + uz);
//                 float r = sigmoid(ar + ur);

//                 // (r*h_prev) for U_h
//                 for(int k=0;k<HIDDEN_DIM;k++){
//                     uh += p.U_h(h,k) * (r * hp[k]);
//                 }

//                 float htil = fast_tanh(ah + uh);
//                 float hnew = (1.0f - z) * hp[h] + z * htil;
//                 // mask: keep previous if padded
//                 hnew = mt * hnew + (1.0f - mt) * hp[h];

//                 // store
//                 H_out[idx3(b,t,h,T,HIDDEN_DIM)] = hnew;
//                 cache.H[idx3(b,t,h,T,HIDDEN_DIM)] = hnew;
//                 cache.Z[idx3(b,t,h,T,HIDDEN_DIM)] = z;
//                 cache.R[idx3(b,t,h,T,HIDDEN_DIM)] = r;
//                 cache.HT[idx3(b,t,h,T,HIDDEN_DIM)] = htil;
//             }

//             // update h_prev row
//             for(int h=0; h<HIDDEN_DIM; h++){
//                 h_prev[size_t(b)*HIDDEN_DIM + h] = H_out[idx3(b,t,h,T,HIDDEN_DIM)];
//             }
//         }
//     }
// }

// struct AttnCache {
//     int B=0,T=0;
//     std::vector<float> H;       // B*T*HIDDEN
//     std::vector<float> M;       // B*T
//     std::vector<float> U;       // B*T*ATTN_DIM
//     std::vector<float> SCORES;  // B*T
//     std::vector<float> ALPHA;   // B*T
// };

// static void attention_forward_batch(
//     const Params& p,
//     const std::vector<float>& H,  // B*T*HIDDEN
//     const std::vector<float>& M,  // B*T
//     int B,int T,
//     std::vector<float>& ctx_out,  // B*HIDDEN
//     AttnCache& cache
// ) {
//     cache = AttnCache();
//     cache.B=B; cache.T=T;
//     cache.H = H;
//     cache.M = M;
//     cache.U.assign(size_t(B)*T*ATTN_DIM, 0.0f);
//     cache.SCORES.assign(size_t(B)*T, 0.0f);

//     // u = tanh(W_att * h)
//     for(int b=0;b<B;b++){
//         for(int t=0;t<T;t++){
//             for(int a=0;a<ATTN_DIM;a++){
//                 float s = 0.0f;
//                 for(int h=0;h<HIDDEN_DIM;h++){
//                     s += p.W_att(a,h) * H[idx3(b,t,h,T,HIDDEN_DIM)];
//                 }
//                 cache.U[idx3(b,t,a,T,ATTN_DIM)] = fast_tanh(s);
//             }
//             // score = v_att dot u
//             float sc=0.0f;
//             for(int a=0;a<ATTN_DIM;a++){
//                 sc += p.v_att(a) * cache.U[idx3(b,t,a,T,ATTN_DIM)];
//             }
//             cache.SCORES[idx2(b,t,T)] = sc;
//         }
//     }

//     // masked softmax scores -> alpha
//     softmax_masked(cache.SCORES, cache.M, B, T, cache.ALPHA);

//     // ctx = sum_t alpha * h
//     ctx_out.assign(size_t(B)*HIDDEN_DIM, 0.0f);
//     for(int b=0;b<B;b++){
//         for(int t=0;t<T;t++){
//             float a = cache.ALPHA[idx2(b,t,T)];
//             for(int h=0;h<HIDDEN_DIM;h++){
//                 ctx_out[size_t(b)*HIDDEN_DIM + h] += a * H[idx3(b,t,h,T,HIDDEN_DIM)];
//             }
//         }
//     }
// }

// static void ms_head_forward(
//     const Params& p,
//     const std::vector<float>& ctx, // B*HIDDEN
//     int B,
//     std::vector<float>& ms_hat,    // B*MS_DIM
//     std::vector<float>& ms_hid     // B*MS_HID (tanh output)
// ) {
//     ms_hid.assign(size_t(B)*MS_HID, 0.0f);
//     ms_hat.assign(size_t(B)*MS_DIM, 0.0f);

//     for(int b=0;b<B;b++){
//         // hid = tanh(W_ms1*ctx + b)
//         for(int j=0;j<MS_HID;j++){
//             float s = p.b_ms1(j);
//             for(int h=0;h<HIDDEN_DIM;h++){
//                 s += p.W_ms1(j,h) * ctx[size_t(b)*HIDDEN_DIM + h];
//             }
//             ms_hid[size_t(b)*MS_HID + j] = fast_tanh(s);
//         }
//         // out = W_ms2*hid + b
//         for(int k=0;k<MS_DIM;k++){
//             float s = p.b_ms2(k);
//             for(int j=0;j<MS_HID;j++){
//                 s += p.W_ms2(k,j) * ms_hid[size_t(b)*MS_HID + j];
//             }
//             ms_hat[size_t(b)*MS_DIM + k] = s;
//         }
//     }
// }

// // ============================================================
// // Backprop pieces for Phase1
// // ============================================================
// static float global_norm(const Params& g) {
//     auto sumsq_mat=[&](const Mat& m){
//         double s=0; for(float v: m.a) s += double(v)*v; return s;
//     };
//     auto sumsq_vec=[&](const Vec& v){
//         double s=0; for(float x: v.a) s += double(x)*x; return s;
//     };
//     double s=0;
//     s += sumsq_mat(g.W_z)+sumsq_mat(g.U_z)+sumsq_vec(g.b_z);
//     s += sumsq_mat(g.W_r)+sumsq_mat(g.U_r)+sumsq_vec(g.b_r);
//     s += sumsq_mat(g.W_h)+sumsq_mat(g.U_h)+sumsq_vec(g.b_h);
//     s += sumsq_mat(g.W_att)+sumsq_vec(g.v_att);
//     s += sumsq_mat(g.W_ms1)+sumsq_vec(g.b_ms1)+sumsq_mat(g.W_ms2)+sumsq_vec(g.b_ms2);
//     s += sumsq_mat(g.W_tok)+sumsq_vec(g.b_tok);
//     s += sumsq_mat(g.W_id)+sumsq_vec(g.b_id);
//     s += sumsq_mat(g.W_w)+sumsq_vec(g.b_w);
//     s += sumsq_mat(g.W_beh)+sumsq_vec(g.b_beh);
//     return float(std::sqrt(s));
// }

// static void clip_grads(Params& g, float max_norm) {
//     float n = global_norm(g);
//     if(n <= max_norm) return;
//     float s = max_norm / (n + 1e-12f);
//     auto scale_mat=[&](Mat& m){ for(float& v: m.a) v*=s; };
//     auto scale_vec=[&](Vec& v){ for(float& x: v.a) x*=s; };

//     scale_mat(g.W_z); scale_mat(g.U_z); scale_vec(g.b_z);
//     scale_mat(g.W_r); scale_mat(g.U_r); scale_vec(g.b_r);
//     scale_mat(g.W_h); scale_mat(g.U_h); scale_vec(g.b_h);
//     scale_mat(g.W_att); scale_vec(g.v_att);
//     scale_mat(g.W_ms1); scale_vec(g.b_ms1); scale_mat(g.W_ms2); scale_vec(g.b_ms2);
//     scale_mat(g.W_tok); scale_vec(g.b_tok);
//     scale_mat(g.W_id); scale_vec(g.b_id);
//     scale_mat(g.W_w);  scale_vec(g.b_w);
//     scale_mat(g.W_beh); scale_vec(g.b_beh);
// }

// // Token CE for one row: logits[V], target -> loss + dlogits
// static float token_ce_one(const std::vector<float>& logits, int target, std::vector<float>& dlogits) {
//     std::vector<float> p;
//     softmax1d(logits, p);
//     float loss = -std::log(p[size_t(target)] + 1e-12f);
//     dlogits = p;
//     dlogits[size_t(target)] -= 1.0f;
//     return loss;
// }

// // ============================================================
// // Phase1 training: MS reconstruction + token scaffold
// // ============================================================
// static Params grads_zeros_like(const Params& p) {
//     return AdamState::zeros_like(p);
// }

// static void extract_batch(const Dataset& ds, const std::vector<int>& indices, int start, int B,
//                           std::vector<float>& Xb, std::vector<float>& Mb, std::vector<int>& Tb,
//                           std::vector<float>& yms, std::vector<float>& ycls) {
//     int bsz = std::min(B, int(indices.size()) - start);
//     Xb.assign(size_t(bsz)*SEQ_LEN*INPUT_DIM, 0.0f);
//     Mb.assign(size_t(bsz)*SEQ_LEN, 0.0f);
//     Tb.assign(size_t(bsz)*SEQ_LEN, 0);
//     yms.assign(size_t(bsz)*MS_DIM, 0.0f);
//     ycls.assign(size_t(bsz), 0.0f);

//     for(int bi=0; bi<bsz; bi++){
//         int n = indices[start + bi];
//         // X, M, TOK
//         for(int t=0;t<SEQ_LEN;t++){
//             Mb[idx2(bi,t,SEQ_LEN)] = ds.M[idx2(n,t,SEQ_LEN)];
//             Tb[idx2(bi,t,SEQ_LEN)] = ds.TOK[idx2(n,t,SEQ_LEN)];
//             for(int d=0; d<INPUT_DIM; d++){
//                 Xb[idx3(bi,t,d,SEQ_LEN,INPUT_DIM)] = ds.X[idx3(n,t,d,SEQ_LEN,INPUT_DIM)];
//             }
//         }
//         // yms
//         for(int k=0;k<MS_DIM;k++){
//             yms[size_t(bi)*MS_DIM + k] = ds.Y_MS[size_t(n)*MS_DIM + k];
//         }
//         ycls[size_t(bi)] = ds.Y_CLS[size_t(n)];
//     }
// }

// static void train_phase1(Params& p, const Dataset& ds) {
//     AdamState opt(p, LR_PHASE1);
//     AdamState::Freeze nofreeze;

//     bool tok_enabled = true;

//     std::vector<int> idx(ds.N);
//     std::iota(idx.begin(), idx.end(), 0);

//     for(int ep=1; ep<=EPOCHS_PHASE1; ep++){
//         // shuffle deterministically per epoch
//         std::shuffle(idx.begin(), idx.end(), std::mt19937(0x1000 + ep));

//         double epoch_loss_sum = 0.0;

//         for(int s=0; s<ds.N; s+=BATCH_SIZE){
//             // batch
//             std::vector<float> Xb, Mb, yms, ycls;
//             std::vector<int> Tb;
//             extract_batch(ds, idx, s, BATCH_SIZE, Xb, Mb, Tb, yms, ycls);
//             int B = int(ycls.size());

//             // forward
//             std::vector<float> H;
//             GRUCache gcache;
//             gru_forward_batch(p, Xb, Mb, B, SEQ_LEN, INPUT_DIM, H, gcache);

//             std::vector<float> ctx;
//             AttnCache acache;
//             attention_forward_batch(p, H, Mb, B, SEQ_LEN, ctx, acache);

//             std::vector<float> ms_hat, ms_hid;
//             ms_head_forward(p, ctx, B, ms_hat, ms_hid);

//             // MS loss only on positives
//             float pos_count = 0.0f;
//             for(int i=0;i<B;i++) if(ycls[size_t(i)] > 0.5f) pos_count += 1.0f;
//             pos_count += 1e-6f;

//             // loss_ms = 0.5 * ||ms_hat - yms||^2 / (pos_count*MS_DIM)
//             float loss_ms = 0.0f;
//             std::vector<float> diff(size_t(B)*MS_DIM, 0.0f);
//             for(int i=0;i<B;i++){
//                 float pm = (ycls[size_t(i)] > 0.5f) ? 1.0f : 0.0f;
//                 for(int k=0;k<MS_DIM;k++){
//                     float d = (ms_hat[size_t(i)*MS_DIM + k] - yms[size_t(i)*MS_DIM + k]) * pm;
//                     diff[size_t(i)*MS_DIM + k] = d;
//                     loss_ms += 0.5f * d*d;
//                 }
//             }
//             loss_ms /= (pos_count * float(MS_DIM));

//             // token scaffold (teacher forcing next token) early epochs
//             float loss_tok = 0.0f;
//             // dH_tok adds to dhs for GRU
//             std::vector<float> dH_tok(size_t(B)*SEQ_LEN*HIDDEN_DIM, 0.0f);
//             Mat dW_tok(p.W_tok.R, p.W_tok.C, 0.0f);
//             Vec db_tok(p.b_tok.N, 0.0f);

//             if(tok_enabled && ep <= TOK_WARMUP_EPOCHS){
//                 float denom = 0.0f;
//                 for(int i=0;i<B;i++){
//                     if(ycls[size_t(i)] <= 0.5f) continue;
//                     for(int t=0;t<SEQ_LEN-1;t++){
//                         float v = Mb[idx2(i,t,SEQ_LEN)] * Mb[idx2(i,t+1,SEQ_LEN)];
//                         if(v>0.5f) denom += 1.0f;
//                     }
//                 }
//                 denom += 1e-6f;

//                 for(int i=0;i<B;i++){
//                     if(ycls[size_t(i)] <= 0.5f) continue;
//                     for(int t=0;t<SEQ_LEN-1;t++){
//                         float v = Mb[idx2(i,t,SEQ_LEN)] * Mb[idx2(i,t+1,SEQ_LEN)];
//                         if(v<=0.5f) continue;

//                         // logits = W_tok * h_t + b
//                         std::vector<float> logits(VOCAB_SIZE, 0.0f);
//                         for(int vj=0; vj<VOCAB_SIZE; vj++){
//                             float s0 = p.b_tok(vj);
//                             for(int h=0; h<HIDDEN_DIM; h++){
//                                 s0 += p.W_tok(vj,h) * H[idx3(i,t,h,SEQ_LEN,HIDDEN_DIM)];
//                             }
//                             logits[vj]=s0;
//                         }
//                         int target = Tb[idx2(i,t+1,SEQ_LEN)];
//                         std::vector<float> dlog;
//                         float li = token_ce_one(logits, target, dlog);
//                         loss_tok += li;

//                         // grads W_tok, b_tok, and dH_tok
//                         for(int vj=0; vj<VOCAB_SIZE; vj++){
//                             db_tok(vj) += dlog[size_t(vj)];
//                             for(int h=0; h<HIDDEN_DIM; h++){
//                                 dW_tok(vj,h) += dlog[size_t(vj)] * H[idx3(i,t,h,SEQ_LEN,HIDDEN_DIM)];
//                             }
//                         }
//                         // dH += dlog^T * W_tok
//                         for(int h=0; h<HIDDEN_DIM; h++){
//                             float sH = 0.0f;
//                             for(int vj=0; vj<VOCAB_SIZE; vj++){
//                                 sH += dlog[size_t(vj)] * p.W_tok(vj,h);
//                             }
//                             dH_tok[idx3(i,t,h,SEQ_LEN,HIDDEN_DIM)] += sH;
//                         }
//                     }
//                 }

//                 loss_tok /= denom;
//                 for(auto& v: dW_tok.a) v /= denom;
//                 for(auto& v: db_tok.a) v /= denom;
//                 for(auto& v: dH_tok)   v /= denom;

//                 if(loss_tok < TOK_STOP_EPS) tok_enabled = false;
//             } else {
//                 tok_enabled = false;
//             }

//             float loss = LAMBDA_MS*loss_ms + (tok_enabled ? (LAMBDA_TOK*loss_tok) : 0.0f);
//             epoch_loss_sum += double(loss) * double(B);

//             // -----------------------
//             // Backprop: MS head -> ctx
//             // -----------------------
//             Params grads = grads_zeros_like(p);
//             grads.W_tok = dW_tok;
//             grads.b_tok = db_tok;

//             // dms = diff / (pos_count*MS_DIM)
//             std::vector<float> dms(size_t(B)*MS_DIM, 0.0f);
//             for(int i=0;i<B;i++){
//                 for(int k=0;k<MS_DIM;k++){
//                     dms[size_t(i)*MS_DIM + k] = diff[size_t(i)*MS_DIM + k] / (pos_count * float(MS_DIM));
//                 }
//             }

//             // grads W_ms2, b_ms2 ; ms_hid = tanh(pre)
//             for(int k=0;k<MS_DIM;k++){
//                 float sb=0.0f;
//                 for(int i=0;i<B;i++) sb += dms[size_t(i)*MS_DIM + k];
//                 grads.b_ms2(k) += sb;
//                 for(int j=0;j<MS_HID;j++){
//                     float sW=0.0f;
//                     for(int i=0;i<B;i++){
//                         sW += dms[size_t(i)*MS_DIM + k] * ms_hid[size_t(i)*MS_HID + j];
//                     }
//                     grads.W_ms2(k,j) += sW;
//                 }
//             }

//             // dms_hid = dms * W_ms2
//             std::vector<float> dms_hid(size_t(B)*MS_HID, 0.0f);
//             for(int i=0;i<B;i++){
//                 for(int j=0;j<MS_HID;j++){
//                     float s=0.0f;
//                     for(int k=0;k<MS_DIM;k++){
//                         s += dms[size_t(i)*MS_DIM + k] * p.W_ms2(k,j);
//                     }
//                     dms_hid[size_t(i)*MS_HID + j] = s;
//                 }
//             }

//             // dpre = dms_hid * (1 - tanh^2)
//             std::vector<float> dpre(size_t(B)*MS_HID, 0.0f);
//             for(int i=0;i<B;i++){
//                 for(int j=0;j<MS_HID;j++){
//                     float h = ms_hid[size_t(i)*MS_HID + j];
//                     dpre[size_t(i)*MS_HID + j] = dms_hid[size_t(i)*MS_HID + j] * (1.0f - h*h);
//                 }
//             }

//             // grads W_ms1, b_ms1 ; dctx = dpre * W_ms1
//             std::vector<float> dctx(size_t(B)*HIDDEN_DIM, 0.0f);
//             for(int j=0;j<MS_HID;j++){
//                 float sb=0.0f;
//                 for(int i=0;i<B;i++) sb += dpre[size_t(i)*MS_HID + j];
//                 grads.b_ms1(j) += sb;
//                 for(int h=0;h<HIDDEN_DIM;h++){
//                     float sW=0.0f;
//                     for(int i=0;i<B;i++){
//                         sW += dpre[size_t(i)*MS_HID + j] * ctx[size_t(i)*HIDDEN_DIM + h];
//                     }
//                     grads.W_ms1(j,h) += sW;
//                 }
//             }
//             for(int i=0;i<B;i++){
//                 for(int h=0;h<HIDDEN_DIM;h++){
//                     float s=0.0f;
//                     for(int j=0;j<MS_HID;j++){
//                         s += dpre[size_t(i)*MS_HID + j] * p.W_ms1(j,h);
//                     }
//                     dctx[size_t(i)*HIDDEN_DIM + h] = s;
//                 }
//             }

//             // -----------------------
//             // Backprop attention -> dH (B*T*HIDDEN)
//             // -----------------------
//             std::vector<float> dH(size_t(B)*SEQ_LEN*HIDDEN_DIM, 0.0f);

//             // dhs from ctx: dhs = alpha * dctx
//             for(int i=0;i<B;i++){
//                 for(int t=0;t<SEQ_LEN;t++){
//                     float a = acache.ALPHA[idx2(i,t,SEQ_LEN)];
//                     for(int h=0;h<HIDDEN_DIM;h++){
//                         dH[idx3(i,t,h,SEQ_LEN,HIDDEN_DIM)] += a * dctx[size_t(i)*HIDDEN_DIM + h];
//                     }
//                 }
//             }

//             // d_alpha = sum_h dctx*h
//             std::vector<float> d_alpha(size_t(B)*SEQ_LEN, 0.0f);
//             for(int i=0;i<B;i++){
//                 for(int t=0;t<SEQ_LEN;t++){
//                     float sA=0.0f;
//                     for(int h=0;h<HIDDEN_DIM;h++){
//                         sA += dctx[size_t(i)*HIDDEN_DIM + h] * H[idx3(i,t,h,SEQ_LEN,HIDDEN_DIM)];
//                     }
//                     d_alpha[idx2(i,t,SEQ_LEN)] = sA;
//                 }
//             }

//             // dscores = alpha*(d_alpha - sum(alpha*d_alpha)) * mask
//             std::vector<float> dscores(size_t(B)*SEQ_LEN, 0.0f);
//             for(int i=0;i<B;i++){
//                 float sum_term=0.0f;
//                 for(int t=0;t<SEQ_LEN;t++){
//                     sum_term += acache.ALPHA[idx2(i,t,SEQ_LEN)] * d_alpha[idx2(i,t,SEQ_LEN)];
//                 }
//                 for(int t=0;t<SEQ_LEN;t++){
//                     float m = Mb[idx2(i,t,SEQ_LEN)];
//                     float a = acache.ALPHA[idx2(i,t,SEQ_LEN)];
//                     dscores[idx2(i,t,SEQ_LEN)] = a * (d_alpha[idx2(i,t,SEQ_LEN)] - sum_term) * m;
//                 }
//             }

//             // grads v_att += sum dscores * u ; du = dscores * v_att
//             for(int a=0;a<ATTN_DIM;a++){
//                 float sv=0.0f;
//                 for(int i=0;i<B;i++){
//                     for(int t=0;t<SEQ_LEN;t++){
//                         sv += dscores[idx2(i,t,SEQ_LEN)] * acache.U[idx3(i,t,a,SEQ_LEN,ATTN_DIM)];
//                     }
//                 }
//                 grads.v_att(a) += sv;
//             }

//             // da = du * (1-u^2)
//             // grads W_att += sum da * h ; and dH += da * W_att
//             for(int i=0;i<B;i++){
//                 for(int t=0;t<SEQ_LEN;t++){
//                     float dsc = dscores[idx2(i,t,SEQ_LEN)];
//                     if(Mb[idx2(i,t,SEQ_LEN)] <= 0.0f) continue;

//                     // compute da[a]
//                     std::vector<float> da(ATTN_DIM, 0.0f);
//                     for(int a=0;a<ATTN_DIM;a++){
//                         float u = acache.U[idx3(i,t,a,SEQ_LEN,ATTN_DIM)];
//                         float du = dsc * p.v_att(a);
//                         da[a] = du * (1.0f - u*u);
//                     }

//                     // W_att grad
//                     for(int a=0;a<ATTN_DIM;a++){
//                         for(int h=0;h<HIDDEN_DIM;h++){
//                             grads.W_att(a,h) += da[a] * H[idx3(i,t,h,SEQ_LEN,HIDDEN_DIM)];
//                         }
//                     }
//                     // dH from da * W_att
//                     for(int h=0;h<HIDDEN_DIM;h++){
//                         float sH=0.0f;
//                         for(int a=0;a<ATTN_DIM;a++){
//                             sH += da[a] * p.W_att(a,h);
//                         }
//                         dH[idx3(i,t,h,SEQ_LEN,HIDDEN_DIM)] += sH;
//                     }
//                 }
//             }

//             // add token scaffold gradient to dH
//             for(size_t ii=0; ii<dH.size(); ii++) dH[ii] += dH_tok[ii];

//             // -----------------------
//             // Backprop GRU (BPTT)
//             // -----------------------
//             // accumulate dW_z, dU_z, db_z etc into grads
//             std::vector<float> dh_next(size_t(B)*HIDDEN_DIM, 0.0f);

//             for(int t=SEQ_LEN-1; t>=0; t--){
//                 for(int i=0;i<B;i++){
//                     float mt = Mb[idx2(i,t,SEQ_LEN)];
//                     // h_prev
//                     const float* hprev = nullptr;
//                     std::vector<float> hprev0;
//                     if(t==0){
//                         hprev0.assign(HIDDEN_DIM, 0.0f);
//                         hprev = hprev0.data();
//                     } else {
//                         hprev = &gcache.H[idx3(i,t-1,0,SEQ_LEN,HIDDEN_DIM)];
//                     }

//                     // current cached
//                     const float* z = &gcache.Z[idx3(i,t,0,SEQ_LEN,HIDDEN_DIM)];
//                     const float* r = &gcache.R[idx3(i,t,0,SEQ_LEN,HIDDEN_DIM)];
//                     const float* htil = &gcache.HT[idx3(i,t,0,SEQ_LEN,HIDDEN_DIM)];

//                     // dh = (dh_next + dH[t]) * mt
//                     std::vector<float> dh(HIDDEN_DIM, 0.0f);
//                     for(int h=0;h<HIDDEN_DIM;h++){
//                         float v = dh_next[size_t(i)*HIDDEN_DIM + h] + dH[idx3(i,t,h,SEQ_LEN,HIDDEN_DIM)];
//                         dh[h] = v * mt;
//                     }

//                     // dh_til = dh * z
//                     std::vector<float> dh_til(HIDDEN_DIM, 0.0f);
//                     std::vector<float> dz(HIDDEN_DIM, 0.0f);
//                     std::vector<float> dh_prev_acc(HIDDEN_DIM, 0.0f);

//                     for(int h=0;h<HIDDEN_DIM;h++){
//                         dh_til[h] = dh[h] * z[h];
//                         dz[h]     = dh[h] * (htil[h] - hprev[h]);
//                         dh_prev_acc[h] = dh[h] * (1.0f - z[h]);
//                     }

//                     // da_h = dh_til * (1-htil^2)
//                     std::vector<float> da_h(HIDDEN_DIM, 0.0f);
//                     for(int h=0;h<HIDDEN_DIM;h++){
//                         da_h[h] = dh_til[h] * (1.0f - htil[h]*htil[h]);
//                         grads.b_h(h) += da_h[h];
//                     }

//                     // dW_h += da_h outer x
//                     const float* xptr = &Xb[idx3(i,t,0,SEQ_LEN,INPUT_DIM)];
//                     for(int h=0;h<HIDDEN_DIM;h++){
//                         for(int d=0; d<INPUT_DIM; d++){
//                             grads.W_h(h,d) += da_h[h] * xptr[d];
//                         }
//                     }
//                     // dU_h += da_h outer (r*hprev)
//                     for(int h=0;h<HIDDEN_DIM;h++){
//                         for(int k=0;k<HIDDEN_DIM;k++){
//                             grads.U_h(h,k) += da_h[h] * (r[h] * hprev[k]);
//                         }
//                     }

//                     // dh_prev += (da_h * U_h) * r
//                     // dr = (da_h * U_h) * hprev
//                     std::vector<float> tmpU(HIDDEN_DIM, 0.0f);
//                     for(int k=0;k<HIDDEN_DIM;k++){
//                         float sU=0.0f;
//                         for(int h=0;h<HIDDEN_DIM;h++){
//                             sU += da_h[h] * p.U_h(h,k);
//                         }
//                         tmpU[k] = sU;
//                         dh_prev_acc[k] += sU * r[k];
//                     }
//                     std::vector<float> dr(HIDDEN_DIM, 0.0f);
//                     for(int k=0;k<HIDDEN_DIM;k++){
//                         dr[k] = tmpU[k] * hprev[k];
//                     }

//                     // da_r = dr * r*(1-r)
//                     std::vector<float> da_r(HIDDEN_DIM, 0.0f);
//                     for(int h=0;h<HIDDEN_DIM;h++){
//                         da_r[h] = dr[h] * r[h] * (1.0f - r[h]);
//                         grads.b_r(h) += da_r[h];
//                     }
//                     // W_r, U_r
//                     for(int h=0;h<HIDDEN_DIM;h++){
//                         for(int d=0; d<INPUT_DIM; d++){
//                             grads.W_r(h,d) += da_r[h] * xptr[d];
//                         }
//                         for(int k=0;k<HIDDEN_DIM;k++){
//                             grads.U_r(h,k) += da_r[h] * hprev[k];
//                         }
//                     }
//                     // dh_prev += da_r * U_r
//                     for(int k=0;k<HIDDEN_DIM;k++){
//                         float sU=0.0f;
//                         for(int h=0;h<HIDDEN_DIM;h++){
//                             sU += da_r[h] * p.U_r(h,k);
//                         }
//                         dh_prev_acc[k] += sU;
//                     }

//                     // da_z = dz * z*(1-z)
//                     std::vector<float> da_z(HIDDEN_DIM, 0.0f);
//                     for(int h=0;h<HIDDEN_DIM;h++){
//                         da_z[h] = dz[h] * z[h] * (1.0f - z[h]);
//                         grads.b_z(h) += da_z[h];
//                     }
//                     // W_z, U_z
//                     for(int h=0;h<HIDDEN_DIM;h++){
//                         for(int d=0; d<INPUT_DIM; d++){
//                             grads.W_z(h,d) += da_z[h] * xptr[d];
//                         }
//                         for(int k=0;k<HIDDEN_DIM;k++){
//                             grads.U_z(h,k) += da_z[h] * hprev[k];
//                         }
//                     }
//                     // dh_prev += da_z * U_z
//                     for(int k=0;k<HIDDEN_DIM;k++){
//                         float sU=0.0f;
//                         for(int h=0;h<HIDDEN_DIM;h++){
//                             sU += da_z[h] * p.U_z(h,k);
//                         }
//                         dh_prev_acc[k] += sU;
//                     }

//                     // write dh_next for this i
//                     for(int h=0;h<HIDDEN_DIM;h++){
//                         dh_next[size_t(i)*HIDDEN_DIM + h] = dh_prev_acc[h];
//                     }
//                 }
//             }

//             // clip + Adam
//             clip_grads(grads, CLIP_NORM);
//             opt.step(p, grads, WEIGHT_DECAY, nofreeze);
//         }

//         if(ep==2 || ep % std::max(1, EPOCHS_PHASE1/10) == 0){
//             cout << "[Phase1] Epoch " << ep << "/" << EPOCHS_PHASE1
//                  << " avg_loss=" << (epoch_loss_sum / double(ds.N))
//                  << " tok_enabled=" << (tok_enabled ? "true":"false")
//                  << endl;
//         }
//     }
// }

// // ============================================================
// // Embeddings for Phase2: ctx + h_last + h_mean
// // ============================================================
// static void compute_embeddings_all(
//     const Params& p,
//     const Dataset& ds,
//     std::vector<float>& ctx_all,   // N*HIDDEN
//     std::vector<float>& hlast_all, // N*HIDDEN
//     std::vector<float>& hmean_all  // N*HIDDEN
// ) {
//     ctx_all.assign(size_t(ds.N)*HIDDEN_DIM, 0.0f);
//     hlast_all.assign(size_t(ds.N)*HIDDEN_DIM, 0.0f);
//     hmean_all.assign(size_t(ds.N)*HIDDEN_DIM, 0.0f);

//     // Process in batches
//     std::vector<int> idx(ds.N);
//     std::iota(idx.begin(), idx.end(), 0);

//     for(int s=0; s<ds.N; s+=BATCH_SIZE){
//         std::vector<float> Xb, Mb, yms, ycls;
//         std::vector<int> Tb;
//         extract_batch(ds, idx, s, BATCH_SIZE, Xb, Mb, Tb, yms, ycls);
//         int B = int(ycls.size());

//         std::vector<float> H;
//         GRUCache gcache;
//         gru_forward_batch(p, Xb, Mb, B, SEQ_LEN, INPUT_DIM, H, gcache);

//         std::vector<float> ctx;
//         AttnCache acache;
//         attention_forward_batch(p, H, Mb, B, SEQ_LEN, ctx, acache);

//         // mean and last over valid mask
//         for(int i=0;i<B;i++){
//             float denom = 0.0f;
//             for(int t=0;t<SEQ_LEN;t++) denom += Mb[idx2(i,t,SEQ_LEN)];
//             denom += 1e-6f;

//             // h_mean
//             for(int h=0;h<HIDDEN_DIM;h++){
//                 float sum=0.0f;
//                 for(int t=0;t<SEQ_LEN;t++){
//                     float m = Mb[idx2(i,t,SEQ_LEN)];
//                     sum += m * H[idx3(i,t,h,SEQ_LEN,HIDDEN_DIM)];
//                 }
//                 hmean_all[size_t(s+i)*HIDDEN_DIM + h] = sum / denom;
//             }

//             // h_last (last valid index)
//             int last = 0;
//             for(int t=0;t<SEQ_LEN;t++){
//                 if(Mb[idx2(i,t,SEQ_LEN)] > 0.5f) last = t;
//             }
//             for(int h=0;h<HIDDEN_DIM;h++){
//                 hlast_all[size_t(s+i)*HIDDEN_DIM + h] = H[idx3(i,last,h,SEQ_LEN,HIDDEN_DIM)];
//             }

//             // ctx
//             for(int h=0;h<HIDDEN_DIM;h++){
//                 ctx_all[size_t(s+i)*HIDDEN_DIM + h] = ctx[size_t(i)*HIDDEN_DIM + h];
//             }
//         }
//     }
// }

// // ============================================================
// // Phase2 training (heads only): ID CE on positives, W CE on positives, Validity BCE on all
// // ============================================================
// static float ce_loss_and_grad(
//     const std::vector<float>& logits, int C,
//     const std::vector<int>& target, const std::vector<float>& mask01,
//     int B,
//     std::vector<float>& dlogits // B*C
// ) {
//     dlogits.assign(size_t(B)*C, 0.0f);
//     int cnt=0;
//     float L=0.0f;

//     for(int i=0;i<B;i++){
//         if(mask01[size_t(i)] <= 0.5f) continue;
//         cnt++;
//         std::vector<float> row(C);
//         for(int c=0;c<C;c++) row[c] = logits[size_t(i)*C + c];
//         std::vector<float> p;
//         softmax1d(row, p);
//         L += -std::log(p[size_t(target[size_t(i)])] + 1e-12f);
//         for(int c=0;c<C;c++){
//             float d = p[size_t(c)];
//             if(c == target[size_t(i)]) d -= 1.0f;
//             dlogits[size_t(i)*C + c] = d;
//         }
//     }
//     if(cnt==0) return 0.0f;
//     float inv = 1.0f / float(cnt);
//     L *= inv;
//     for(float& v: dlogits) v *= inv;
//     return L;
// }

// static float bce_loss_and_grad(
//     const std::vector<float>& logits, // B
//     const std::vector<float>& y,      // B
//     int B,
//     std::vector<float>& dlog // B
// ) {
//     dlog.assign(size_t(B), 0.0f);
//     float L=0.0f;
//     for(int i=0;i<B;i++){
//         float p = sigmoid(logits[size_t(i)]);
//         float yi = y[size_t(i)];
//         float eps=1e-8f;
//         float loss = -(POS_WEIGHT*yi*std::log(p+eps) + (1.0f-yi)*std::log(1.0f-p+eps));
//         L += loss;

//         float dl = (p - yi);
//         if(yi > 0.5f) dl *= POS_WEIGHT;
//         dlog[size_t(i)] = dl;
//     }
//     L /= float(B);
//     for(float& v: dlog) v /= float(B);
//     return L;
// }

// static void train_phase2(Params& p, const Dataset& ds) {
//     // precompute embeddings from frozen backbone
//     std::vector<float> ctx_all, hlast_all, hmean_all;
//     compute_embeddings_all(p, ds, ctx_all, hlast_all, hmean_all);

//     AdamState opt(p, LR_PHASE2_BASE);

//     // freeze everything except heads
//     AdamState::Freeze fr;
//     fr.W_z=fr.U_z=fr.b_z=true;
//     fr.W_r=fr.U_r=fr.b_r=true;
//     fr.W_h=fr.U_h=fr.b_h=true;
//     fr.W_att=fr.v_att=true;
//     fr.W_ms1=fr.b_ms1=fr.W_ms2=fr.b_ms2=true;
//     fr.W_tok=fr.b_tok=true;
//     // Trainable:
//     fr.W_id=fr.b_id=false;
//     fr.W_w=fr.b_w=false;
//     fr.W_beh=fr.b_beh=false;

//     std::vector<int> idx(ds.N);
//     std::iota(idx.begin(), idx.end(), 0);

//     for(int ep=1; ep<=EPOCHS_PHASE2; ep++){
//         // lr decay
//         float lr = LR_PHASE2_BASE * std::pow(0.98f, float(ep)/30.0f);
//         opt.lr = lr;

//         std::shuffle(idx.begin(), idx.end(), std::mt19937(0x9000 + ep));

//         double epoch_loss_sum=0.0;

//         for(int s=0; s<ds.N; s+=BATCH_SIZE){
//             int B = std::min(BATCH_SIZE, ds.N - s);

//             // batch views
//             std::vector<float> cb(size_t(B)*HIDDEN_DIM,0.0f);
//             std::vector<float> hl(size_t(B)*HIDDEN_DIM,0.0f);
//             std::vector<float> hm(size_t(B)*HIDDEN_DIM,0.0f);
//             std::vector<float> yb(size_t(B),0.0f);
//             std::vector<float> pos(size_t(B),0.0f);
//             std::vector<int> tid(size_t(B),0), tw(size_t(B),0), claim(size_t(B),0), expw(size_t(B),0);

//             for(int i=0;i<B;i++){
//                 int n = idx[s+i];
//                 for(int h=0;h<HIDDEN_DIM;h++){
//                     cb[size_t(i)*HIDDEN_DIM+h] = ctx_all[size_t(n)*HIDDEN_DIM+h];
//                     hl[size_t(i)*HIDDEN_DIM+h] = hlast_all[size_t(n)*HIDDEN_DIM+h];
//                     hm[size_t(i)*HIDDEN_DIM+h] = hmean_all[size_t(n)*HIDDEN_DIM+h];
//                 }
//                 yb[size_t(i)] = ds.Y_CLS[size_t(n)];
//                 pos[size_t(i)] = (yb[size_t(i)]>0.5f) ? 1.0f : 0.0f;
//                 tid[size_t(i)] = ds.TRUE_ID[n];
//                 tw[size_t(i)]  = ds.TRUE_W[n];
//                 claim[size_t(i)] = ds.CLAIM_ID[n];
//                 expw[size_t(i)]  = ds.EXPECT_W[n];
//             }

//             // logits_id = cb * W_id^T + b_id
//             std::vector<float> logits_id(size_t(B)*N_IDENTITIES, 0.0f);
//             for(int i=0;i<B;i++){
//                 for(int c=0;c<N_IDENTITIES;c++){
//                     float s0 = p.b_id(c);
//                     for(int h=0;h<HIDDEN_DIM;h++){
//                         s0 += p.W_id(c,h) * cb[size_t(i)*HIDDEN_DIM + h];
//                     }
//                     logits_id[size_t(i)*N_IDENTITIES + c] = s0;
//                 }
//             }

//             // feat_w = [cb, hl, hm] size 3*HIDDEN
//             std::vector<float> feat_w(size_t(B)*3*HIDDEN_DIM, 0.0f);
//             for(int i=0;i<B;i++){
//                 for(int h=0;h<HIDDEN_DIM;h++){
//                     feat_w[size_t(i)*3*HIDDEN_DIM + h] = cb[size_t(i)*HIDDEN_DIM+h];
//                     feat_w[size_t(i)*3*HIDDEN_DIM + (HIDDEN_DIM+h)] = hl[size_t(i)*HIDDEN_DIM+h];
//                     feat_w[size_t(i)*3*HIDDEN_DIM + (2*HIDDEN_DIM+h)] = hm[size_t(i)*HIDDEN_DIM+h];
//                 }
//             }

//             // logits_w = feat_w * W_w^T + b_w
//             std::vector<float> logits_w(size_t(B)*N_WINDOWS_PER_ID, 0.0f);
//             for(int i=0;i<B;i++){
//                 for(int c=0;c<N_WINDOWS_PER_ID;c++){
//                     float s0 = p.b_w(c);
//                     for(int j=0;j<3*HIDDEN_DIM;j++){
//                         s0 += p.W_w(c,j) * feat_w[size_t(i)*3*HIDDEN_DIM + j];
//                     }
//                     logits_w[size_t(i)*N_WINDOWS_PER_ID + c] = s0;
//                 }
//             }

//             // prob_id/prob_w -> p_id_claimed, p_w_expected
//             std::vector<float> p_id_claimed(size_t(B),0.0f);
//             std::vector<float> p_w_expected(size_t(B),0.0f);

//             for(int i=0;i<B;i++){
//                 // id
//                 std::vector<float> row_id(N_IDENTITIES);
//                 for(int c=0;c<N_IDENTITIES;c++) row_id[c] = logits_id[size_t(i)*N_IDENTITIES+c];
//                 std::vector<float> pid;
//                 softmax1d(row_id, pid);
//                 p_id_claimed[size_t(i)] = pid[size_t(claim[size_t(i)])];

//                 // w
//                 std::vector<float> row_w(N_WINDOWS_PER_ID);
//                 for(int c=0;c<N_WINDOWS_PER_ID;c++) row_w[c] = logits_w[size_t(i)*N_WINDOWS_PER_ID+c];
//                 std::vector<float> pw;
//                 softmax1d(row_w, pw);
//                 p_w_expected[size_t(i)] = pw[size_t(expw[size_t(i)])];
//             }

//             // validity input: [cb, cid, ew, pid, pw] -> dim HIDDEN+4
//             std::vector<float> vb_in(size_t(B)*(HIDDEN_DIM+4), 0.0f);
//             std::vector<float> logits_v(size_t(B), 0.0f);

//             for(int i=0;i<B;i++){
//                 float cid = float(claim[size_t(i)]) / float(std::max(1, N_IDENTITIES-1));
//                 float ew  = float(expw[size_t(i)])  / float(std::max(1, N_WINDOWS_PER_ID-1));
//                 for(int h=0;h<HIDDEN_DIM;h++){
//                     vb_in[size_t(i)*(HIDDEN_DIM+4) + h] = cb[size_t(i)*HIDDEN_DIM + h];
//                 }
//                 vb_in[size_t(i)*(HIDDEN_DIM+4) + HIDDEN_DIM + 0] = cid;
//                 vb_in[size_t(i)*(HIDDEN_DIM+4) + HIDDEN_DIM + 1] = ew;
//                 vb_in[size_t(i)*(HIDDEN_DIM+4) + HIDDEN_DIM + 2] = p_id_claimed[size_t(i)];
//                 vb_in[size_t(i)*(HIDDEN_DIM+4) + HIDDEN_DIM + 3] = p_w_expected[size_t(i)];

//                 float s0 = p.b_beh(0);
//                 for(int j=0;j<HIDDEN_DIM+4;j++){
//                     s0 += p.W_beh(0,j) * vb_in[size_t(i)*(HIDDEN_DIM+4) + j];
//                 }
//                 logits_v[size_t(i)] = s0;
//             }

//             // losses + grads
//             std::vector<float> dlog_id, dlog_w, dlog_v;
//             float loss_id = ce_loss_and_grad(logits_id, N_IDENTITIES, tid, pos, B, dlog_id);
//             float loss_w  = ce_loss_and_grad(logits_w, N_WINDOWS_PER_ID, tw, pos, B, dlog_w);
//             float loss_v  = bce_loss_and_grad(logits_v, yb, B, dlog_v);

//             float loss = LAMBDA_ID*loss_id + LAMBDA_W*loss_w + LAMBDA_BCE*loss_v;
//             epoch_loss_sum += double(loss) * double(B);

//             Params grads = grads_zeros_like(p);

//             // W_id grad: dlog_id^T @ cb ; b_id grad sum
//             for(int c=0;c<N_IDENTITIES;c++){
//                 float sb=0.0f;
//                 for(int i=0;i<B;i++) sb += dlog_id[size_t(i)*N_IDENTITIES + c];
//                 grads.b_id(c) += sb;

//                 for(int h=0;h<HIDDEN_DIM;h++){
//                     float sW=0.0f;
//                     for(int i=0;i<B;i++){
//                         sW += dlog_id[size_t(i)*N_IDENTITIES + c] * cb[size_t(i)*HIDDEN_DIM + h];
//                     }
//                     grads.W_id(c,h) += sW;
//                 }
//             }

//             // W_w grad: dlog_w^T @ feat_w ; b_w
//             for(int c=0;c<N_WINDOWS_PER_ID;c++){
//                 float sb=0.0f;
//                 for(int i=0;i<B;i++) sb += dlog_w[size_t(i)*N_WINDOWS_PER_ID + c];
//                 grads.b_w(c) += sb;

//                 for(int j=0;j<3*HIDDEN_DIM;j++){
//                     float sW=0.0f;
//                     for(int i=0;i<B;i++){
//                         sW += dlog_w[size_t(i)*N_WINDOWS_PER_ID + c] * feat_w[size_t(i)*3*HIDDEN_DIM + j];
//                     }
//                     grads.W_w(c,j) += sW;
//                 }
//             }

//             // W_beh grad: sum_i dlog_v[i]*vb_in[i] ; b_beh
//             float sb=0.0f;
//             for(int i=0;i<B;i++) sb += dlog_v[size_t(i)];
//             grads.b_beh(0) += sb;

//             for(int j=0;j<HIDDEN_DIM+4;j++){
//                 float sW=0.0f;
//                 for(int i=0;i<B;i++){
//                     sW += dlog_v[size_t(i)] * vb_in[size_t(i)*(HIDDEN_DIM+4) + j];
//                 }
//                 grads.W_beh(0,j) += sW;
//             }

//             clip_grads(grads, CLIP_NORM);
//             opt.step(p, grads, WEIGHT_DECAY, fr);
//         }

//         if(ep==2 || ep % std::max(1, EPOCHS_PHASE2/10) == 0){
//             cout << "[Phase2] Epoch " << ep << "/" << EPOCHS_PHASE2
//                  << " avg_loss=" << (epoch_loss_sum / double(ds.N))
//                  << " lr=" << lr
//                  << endl;
//         }
//     }
// }

// // ============================================================
// // Save / Load model (binary, simple & robust)
// // ============================================================
// static void write_u32(std::ofstream& f, uint32_t x){ f.write(reinterpret_cast<const char*>(&x), sizeof(x)); }
// static void write_u64(std::ofstream& f, uint64_t x){ f.write(reinterpret_cast<const char*>(&x), sizeof(x)); }
// // static void write_f32(std::ofstream& f, float x){ f.write(reinterpret_cast<const char*>(&x), sizeof(x)); }

// static uint32_t read_u32(std::ifstream& f){ uint32_t x; f.read(reinterpret_cast<char*>(&x), sizeof(x)); return x; }
// static uint64_t read_u64(std::ifstream& f){ uint64_t x; f.read(reinterpret_cast<char*>(&x), sizeof(x)); return x; }
// // static float read_f32(std::ifstream& f){ float x; f.read(reinterpret_cast<char*>(&x), sizeof(x)); return x; }

// static void write_mat(std::ofstream& f, const Mat& m){
//     write_u32(f, (uint32_t)m.R);
//     write_u32(f, (uint32_t)m.C);
//     write_u64(f, (uint64_t)m.a.size());
//     f.write(reinterpret_cast<const char*>(m.a.data()), std::streamsize(m.a.size()*sizeof(float)));
// }
// static void write_vec(std::ofstream& f, const Vec& v){
//     write_u32(f, (uint32_t)v.N);
//     write_u64(f, (uint64_t)v.a.size());
//     f.write(reinterpret_cast<const char*>(v.a.data()), std::streamsize(v.a.size()*sizeof(float)));
// }

// static void read_mat(std::ifstream& f, Mat& m){
//     uint32_t R = read_u32(f);
//     uint32_t C = read_u32(f);
//     uint64_t sz = read_u64(f);
//     m.R = int(R); m.C = int(C);
//     m.a.assign(size_t(sz), 0.0f);
//     f.read(reinterpret_cast<char*>(m.a.data()), std::streamsize(m.a.size()*sizeof(float)));
// }
// static void read_vec(std::ifstream& f, Vec& v){
//     uint32_t N = read_u32(f);
//     uint64_t sz = read_u64(f);
//     v.N = int(N);
//     v.a.assign(size_t(sz), 0.0f);
//     f.read(reinterpret_cast<char*>(v.a.data()), std::streamsize(v.a.size()*sizeof(float)));
// }

// static bool save_model(const std::string& path, const Params& p){
//     std::ofstream f(path, std::ios::binary);
//     if(!f) return false;

//     // header
//     write_u32(f, 0x57414C4Bu); // 'WALK' magic-ish
//     write_u32(f, 1u);          // version

//     // store a few constants for sanity
//     write_u32(f, (uint32_t)VOCAB_SIZE);
//     write_u32(f, (uint32_t)MS_DIM);
//     write_u32(f, (uint32_t)SEQ_LEN);
//     write_u32(f, (uint32_t)N_IDENTITIES);
//     write_u32(f, (uint32_t)N_WINDOWS_PER_ID);
//     write_u32(f, (uint32_t)HIDDEN_DIM);
//     write_u32(f, (uint32_t)ATTN_DIM);
//     write_u32(f, (uint32_t)MS_HID);
//     write_u32(f, (uint32_t)INPUT_DIM);

//     // params
//     write_mat(f,p.W_z); write_mat(f,p.U_z); write_vec(f,p.b_z);
//     write_mat(f,p.W_r); write_mat(f,p.U_r); write_vec(f,p.b_r);
//     write_mat(f,p.W_h); write_mat(f,p.U_h); write_vec(f,p.b_h);

//     write_mat(f,p.W_att); write_vec(f,p.v_att);

//     write_mat(f,p.W_ms1); write_vec(f,p.b_ms1);
//     write_mat(f,p.W_ms2); write_vec(f,p.b_ms2);

//     write_mat(f,p.W_tok); write_vec(f,p.b_tok);

//     write_mat(f,p.W_id); write_vec(f,p.b_id);
//     write_mat(f,p.W_w);  write_vec(f,p.b_w);

//     write_mat(f,p.W_beh); write_vec(f,p.b_beh);

//     return true;
// }

// static bool load_model(const std::string& path, Params& p){
//     std::ifstream f(path, std::ios::binary);
//     if(!f) return false;

//     uint32_t magic = read_u32(f);
//     uint32_t ver   = read_u32(f);
//     if(magic != 0x57414C4Bu || ver != 1u) return false;

//     // constants sanity
//     uint32_t v_voc = read_u32(f);
//     uint32_t v_msd = read_u32(f);
//     uint32_t v_seq = read_u32(f);
//     uint32_t v_nid = read_u32(f);
//     uint32_t v_nw  = read_u32(f);
//     uint32_t v_hid = read_u32(f);
//     uint32_t v_att = read_u32(f);
//     uint32_t v_msh = read_u32(f);
//     uint32_t v_in  = read_u32(f);
//     (void)v_voc; (void)v_msd; (void)v_seq; (void)v_nid; (void)v_nw; (void)v_hid; (void)v_att; (void)v_msh; (void)v_in;

//     read_mat(f,p.W_z); read_mat(f,p.U_z); read_vec(f,p.b_z);
//     read_mat(f,p.W_r); read_mat(f,p.U_r); read_vec(f,p.b_r);
//     read_mat(f,p.W_h); read_mat(f,p.U_h); read_vec(f,p.b_h);

//     read_mat(f,p.W_att); read_vec(f,p.v_att);

//     read_mat(f,p.W_ms1); read_vec(f,p.b_ms1);
//     read_mat(f,p.W_ms2); read_vec(f,p.b_ms2);

//     read_mat(f,p.W_tok); read_vec(f,p.b_tok);

//     read_mat(f,p.W_id); read_vec(f,p.b_id);
//     read_mat(f,p.W_w);  read_vec(f,p.b_w);

//     read_mat(f,p.W_beh); read_vec(f,p.b_beh);

//     return true;
// }

// // ============================================================
// // Explainable verify (single sample)
// // ============================================================
// struct VerifyOut {
//     bool ok=false;
//     float p_valid=0;
//     int id_pred=-1;
//     int w_pred=-1;
//     float pid=0;
//     float pw=0;
//     float l2ms=-1;
// };

// static VerifyOut verify_chain(
//     const Params& p,
//     const std::vector<int>& tokens,
//     const std::vector<float>& meas,
//     int claimed_id,
//     int expected_w,
//     const std::vector<float>* true_ms // optional
// ){
//     // build X (single)
//     Mat X;
//     build_X_backbone(tokens, meas, X);

//     std::vector<float> Xb(size_t(1)*SEQ_LEN*INPUT_DIM, 0.0f);
//     std::vector<float> Mb(size_t(1)*SEQ_LEN, 1.0f);
//     for(int t=0;t<SEQ_LEN;t++){
//         for(int d=0; d<INPUT_DIM; d++){
//             Xb[idx3(0,t,d,SEQ_LEN,INPUT_DIM)] = X(t,d);
//         }
//     }

//     // forward
//     std::vector<float> H;
//     GRUCache gcache;
//     gru_forward_batch(p, Xb, Mb, 1, SEQ_LEN, INPUT_DIM, H, gcache);

//     std::vector<float> ctx;
//     AttnCache acache;
//     attention_forward_batch(p, H, Mb, 1, SEQ_LEN, ctx, acache);

//     std::vector<float> ms_hat, ms_hid;
//     ms_head_forward(p, ctx, 1, ms_hat, ms_hid);

//     // logits_id
//     std::vector<float> logits_id(N_IDENTITIES, 0.0f);
//     for(int c=0;c<N_IDENTITIES;c++){
//         float s0 = p.b_id(c);
//         for(int h=0;h<HIDDEN_DIM;h++){
//             s0 += p.W_id(c,h) * ctx[h];
//         }
//         logits_id[c]=s0;
//     }
//     std::vector<float> prob_id;
//     softmax1d(logits_id, prob_id);
//     int id_pred = int(std::max_element(prob_id.begin(), prob_id.end()) - prob_id.begin());
//     float pid = prob_id[size_t(claimed_id)];

//     // logits_w
//     std::vector<float> feat_w(3*HIDDEN_DIM, 0.0f);
//     // h_last, h_mean
//     std::vector<float> hmean(HIDDEN_DIM,0.0f), hlast(HIDDEN_DIM,0.0f);
//     // mean
//     float denom = 0.0f;
//     for(int t=0;t<SEQ_LEN;t++) denom += Mb[idx2(0,t,SEQ_LEN)];
//     denom += 1e-6f;
//     for(int h=0;h<HIDDEN_DIM;h++){
//         float sum=0.0f;
//         for(int t=0;t<SEQ_LEN;t++){
//             sum += Mb[idx2(0,t,SEQ_LEN)] * H[idx3(0,t,h,SEQ_LEN,HIDDEN_DIM)];
//         }
//         hmean[h] = sum/denom;
//     }
//     // last
//     int last=SEQ_LEN-1;
//     for(int h=0;h<HIDDEN_DIM;h++) hlast[h] = H[idx3(0,last,h,SEQ_LEN,HIDDEN_DIM)];

//     for(int h=0;h<HIDDEN_DIM;h++){
//         feat_w[h] = ctx[h];
//         feat_w[HIDDEN_DIM+h] = hlast[h];
//         feat_w[2*HIDDEN_DIM+h] = hmean[h];
//     }

//     std::vector<float> logits_w(N_WINDOWS_PER_ID, 0.0f);
//     for(int c=0;c<N_WINDOWS_PER_ID;c++){
//         float s0 = p.b_w(c);
//         for(int j=0;j<3*HIDDEN_DIM;j++){
//             s0 += p.W_w(c,j) * feat_w[j];
//         }
//         logits_w[c]=s0;
//     }
//     std::vector<float> prob_w;
//     softmax1d(logits_w, prob_w);
//     int w_pred = int(std::max_element(prob_w.begin(), prob_w.end()) - prob_w.begin());
//     float pw = prob_w[size_t(expected_w)];

//     // validity head
//     std::vector<float> vb_in(HIDDEN_DIM+4, 0.0f);
//     for(int h=0;h<HIDDEN_DIM;h++) vb_in[h]=ctx[h];
//     float cid = float(claimed_id)/float(std::max(1,N_IDENTITIES-1));
//     float ew  = float(expected_w)/float(std::max(1,N_WINDOWS_PER_ID-1));
//     vb_in[HIDDEN_DIM+0]=cid;
//     vb_in[HIDDEN_DIM+1]=ew;
//     vb_in[HIDDEN_DIM+2]=pid;
//     vb_in[HIDDEN_DIM+3]=pw;

//     float logit_v = p.b_beh(0);
//     for(int j=0;j<HIDDEN_DIM+4;j++) logit_v += p.W_beh(0,j)*vb_in[j];
//     float p_valid = sigmoid(logit_v);

//     VerifyOut out;
//     out.id_pred=id_pred;
//     out.w_pred=w_pred;
//     out.pid=pid;
//     out.pw=pw;
//     out.p_valid=p_valid;

//     bool c1 = (p_valid >= THRESH_P_VALID);
//     bool c2 = (id_pred == claimed_id);
//     bool c3 = (w_pred == expected_w);
//     bool c4 = (pid >= PID_MIN);
//     bool c5 = (pw >= PW_MIN);
//     out.ok = (c1 && c2 && c3 && c4 && c5);

//     if(true_ms){
//         std::vector<float> ms_true = *true_ms;
//         std::vector<float> ms_hat_v(MS_DIM);
//         for(int k=0;k<MS_DIM;k++) ms_hat_v[k] = ms_hat[size_t(k)];
//         out.l2ms = l2_vec(ms_hat_v, ms_true);
//     }
//     return out;
// }

// static void print_case(const std::string& title, const VerifyOut& r, int claimed_id, int expected_w) {
//     cout << "\n=== " << title << " ===\n";
//     cout << "CLAIMED: PeerID=" << claimed_id << " ExpectedWindow=" << expected_w << "\n";
//     cout << "PRED:    id_pred=" << r.id_pred << " w_pred=" << r.w_pred << "\n";
//     cout << "SCORES:  p_valid=" << r.p_valid << " pid=" << r.pid << " pw=" << r.pw << "\n";
//     if(r.l2ms >= 0) cout << "RECON:   L2(MS_hat,MS_true)=" << r.l2ms << "\n";
//     cout << "GATES:   "
//          << (r.p_valid >= THRESH_P_VALID ? "pV " : "pVx ")
//          << (r.id_pred == claimed_id ? "id " : "idx ")
//          << (r.w_pred == expected_w ? "w " : "wx ")
//          << (r.pid >= PID_MIN ? "pid " : "pidx ")
//          << (r.pw >= PW_MIN ? "pw " : "pwx ")
//          << "\n";
//     cout << "FINAL:   OK=" << (r.ok ? "true":"false") << "\n";
// }

// // ============================================================
// // Main demo: init -> dataset -> train -> save -> load -> timed inference
// // ============================================================
// int main() {
//     cout << "WARL0K PIM C++ Demo (GRU+Attention, 2-phase, save/load, timing/sizing)\n";
//     cout << "PN PILOT ENABLED amp=" << PILOT_AMP << " chips=" << SEQ_LEN << "\n\n";

//     // init global MS_all and A_base
//     {
//         XorShift32 rng(0xDEADBEEFu);
//         MS_all = Mat(N_IDENTITIES, MS_DIM, 0.0f);
//         for(int i=0;i<N_IDENTITIES;i++){
//             for(int k=0;k<MS_DIM;k++){
//                 // uniform(-1,1)
//                 float u = rng.next_f01();
//                 MS_all(i,k) = 2.0f*u - 1.0f;
//             }
//         }
//         A_base = Mat(SEQ_LEN, MS_DIM, 0.0f);
//         for(int t=0;t<SEQ_LEN;t++){
//             for(int k=0;k<MS_DIM;k++){
//                 A_base(t,k) = 0.8f * rng.next_norm();
//             }
//         }
//     }

//     // dataset
//     Dataset ds = build_dataset();
//     int positives = 0;
//     for(float y: ds.Y_CLS) if(y > 0.5f) positives++;
//     cout << "All samples: " << ds.N
//          << " X shape: (" << ds.N << "," << SEQ_LEN << "," << INPUT_DIM << ")"
//          << " Positives: " << positives << "\n\n";

//     // init model
//     Params p = init_model(INPUT_DIM, 0xC0FFEEu);
//     cout << "Model params: " << p.param_count()
//          << " floats, size ~" << (double(p.bytes())/1024.0) << " KB\n";

//     // training time
//     double t0 = now_seconds();
//     train_phase1(p, ds);
//     train_phase2(p, ds);
//     double t1 = now_seconds();
//     cout << "\nTRAINING TIME (total): " << (t1 - t0) << " sec\n";

//     // save model
//     const std::string model_path = "warlok_pim_model.bin";
//     if(!save_model(model_path, p)){
//         cout << "ERROR: failed to save model\n";
//         return 1;
//     }
//     cout << "Saved model -> " << model_path << "\n";

//     // load model into p2
//     Params p2;
//     if(!load_model(model_path, p2)){
//         cout << "ERROR: failed to load model\n";
//         return 1;
//     }
//     cout << "Loaded model OK. Params floats=" << p2.param_count()
//          << " size ~" << (double(p2.bytes())/1024.0) << " KB\n";

//     // pick eval case
//     int id_eval = 0;
//     int w_eval  = 5;
//     std::vector<float> ms_eval(MS_DIM);
//     for(int k=0;k<MS_DIM;k++) ms_eval[k] = MS_all(id_eval,k);

//     int g_true = id_eval * N_WINDOWS_PER_ID + w_eval;
//     std::vector<int> toks;
//     std::vector<float> meas;
//     generate_os_chain(ms_eval, g_true, toks, meas);

//     // Timed inference (run N times)
//     const int INFER_ITERS = 200;
//     double ti0 = now_seconds();
//     VerifyOut last;
//     for(int i=0;i<INFER_ITERS;i++){
//         last = verify_chain(p2, toks, meas, id_eval, w_eval, &ms_eval);
//     }
//     double ti1 = now_seconds();
//     double per = (ti1 - ti0) / double(INFER_ITERS);
//     cout << "\nINFERENCE TIME: total=" << (ti1-ti0) << " sec"
//          << " per_call=" << per*1e6 << " us (" << per*1e3 << " ms)\n";

//     // Show cases (legit + tamper variants)
//     {
//         auto r = verify_chain(p2, toks, meas, id_eval, w_eval, &ms_eval);
//         print_case("LEGIT (intact chain)", r, id_eval, w_eval);

//         // shuffled
//         std::vector<int> idxs(SEQ_LEN);
//         std::iota(idxs.begin(), idxs.end(), 0);
//         std::shuffle(idxs.begin(), idxs.end(), std::mt19937(0x7777));
//         std::vector<int> toks_s(SEQ_LEN);
//         std::vector<float> meas_s(SEQ_LEN);
//         for(int t=0;t<SEQ_LEN;t++){ toks_s[t]=toks[idxs[t]]; meas_s[t]=meas[idxs[t]]; }
//         r = verify_chain(p2, toks_s, meas_s, id_eval, w_eval, &ms_eval);
//         print_case("TAMPER: SHUFFLED (replay-ish)", r, id_eval, w_eval);

//         // truncated (half, rest zeros)
//         int Ltr = SEQ_LEN/2;
//         std::vector<int> toks_t(SEQ_LEN, 0);
//         std::vector<float> meas_t(SEQ_LEN, 0.0f);
//         for(int t=0;t<Ltr;t++){ toks_t[t]=toks[t]; meas_t[t]=meas[t]; }
//         r = verify_chain(p2, toks_t, meas_t, id_eval, w_eval, &ms_eval);
//         print_case("TAMPER: TRUNCATED (dropped steps)", r, id_eval, w_eval);

//         // wrong window
//         int wrong_w = (w_eval + 7) % N_WINDOWS_PER_ID;
//         int g_wrong = id_eval * N_WINDOWS_PER_ID + wrong_w;
//         std::vector<int> toks_w;
//         std::vector<float> meas_w;
//         generate_os_chain(ms_eval, g_wrong, toks_w, meas_w);
//         r = verify_chain(p2, toks_w, meas_w, id_eval, w_eval, &ms_eval);
//         print_case("TAMPER: WRONG WINDOW (counter drift)", r, id_eval, w_eval);

//         // wrong identity
//         int other_id = (id_eval + 1) % N_IDENTITIES;
//         int other_w  = 13;
//         int g_other  = other_id * N_WINDOWS_PER_ID + other_w;
//         std::vector<float> ms_other(MS_DIM);
//         for(int k=0;k<MS_DIM;k++) ms_other[k] = MS_all(other_id,k);
//         std::vector<int> toks_o;
//         std::vector<float> meas_o;
//         generate_os_chain(ms_other, g_other, toks_o, meas_o);
//         r = verify_chain(p2, toks_o, meas_o, id_eval, w_eval, &ms_eval);
//         print_case("TAMPER: WRONG IDENTITY (impersonation)", r, id_eval, w_eval);
//     }

//     cout << "\nDone.\n";
//     return 0;
// }

// WARLOK PIM GRU+Attention C++ inference
// ------------------------------------------------------------
// warlok_pim.cpp
// WARL0K PIM C++ Demo (GRU+Attention, 2-phase, save/load, timing/sizing)
// Improvements:
// - No std::mt19937 (custom deterministic shuffle)
// - No unused helpers (clean under -Werror)
// - Phase2 stabilized (lower LR + optional embedding refresh per epoch)
// - Hard chain-break for out-of-range (claimed_id/expected_w)
// - PN pilot correlation gate (early tamper detect)
// - Prints model file size (no <filesystem>, uses ifstream|ate)
//
// Build:
//   g++ -O2 -std=c++17 -Wall -Wextra -Werror warlok_pim.cpp -o pim
// Run:
//   ./pim

#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <limits>
#include <random>

using std::cout;
using std::endl;

// ============================================================
// Config
// ============================================================
static constexpr int VOCAB_SIZE       = 16;
static constexpr int MS_DIM           = 8;
static constexpr int SEQ_LEN          = 20;

static constexpr int N_IDENTITIES     = 2;
static constexpr int N_WINDOWS_PER_ID = 48;

static constexpr int HIDDEN_DIM       = 64;
static constexpr int ATTN_DIM         = 32;
static constexpr int MS_HID           = 32;

static constexpr int BATCH_SIZE       = 32;

static constexpr int EPOCHS_PHASE1    = 90;
static constexpr int EPOCHS_PHASE2    = 100;

static constexpr float LR_PHASE1      = 0.006f;
// Lowered to reduce Phase2 oscillations
static constexpr float LR_PHASE2_BASE = 0.008f;

static constexpr float CLIP_NORM      = 5.0f;
static constexpr float WEIGHT_DECAY   = 1e-4f;

// Phase1 losses
static constexpr float LAMBDA_MS      = 1.0f;
static constexpr float LAMBDA_TOK     = 0.10f;
static constexpr float TOK_STOP_EPS   = 0.25f;
static constexpr int   TOK_WARMUP_EPOCHS = 60;

// Phase2 losses
static constexpr float LAMBDA_ID      = 1.0f;
static constexpr float LAMBDA_W       = 1.0f;
static constexpr float LAMBDA_BCE     = 1.0f;
static constexpr float POS_WEIGHT     = 10.0f;

// Strong accept thresholds (demo gating)
static constexpr float THRESH_P_VALID = 0.80f;
static constexpr float PID_MIN        = 0.70f;
// static constexpr float PW_MIN         = 0.70f;
static constexpr float PW_MIN = 0.40f;

// static constexpr float W_MARGIN_MIN = 1.0f;   // in logit units, tune 0.6–1.5
// // or
// static constexpr float W_RATIO_MIN  = 3.0f;   // pw_top / pw_second

// Window pilot (PN watermark)
static constexpr float PILOT_AMP      = 0.55f;
// Correlation threshold for PN pilot gate (tune 0.10–0.30)
static constexpr float PILOT_CORR_MIN = 0.02f;

// Backbone input dim: one-hot vocab + meas + time
static constexpr int INPUT_DIM        = VOCAB_SIZE + 2;

// ============================================================
// Small utilities
// ============================================================
static inline float sigmoid(float x) {
    if (x >= 0) {
        float z = std::exp(-x);
        return 1.0f / (1.0f + z);
    } else {
        float z = std::exp(x);
        return z / (1.0f + z);
    }
}

static inline float fast_tanh(float x) {
    return std::tanh(x);
}

static inline float clampf(float x, float lo, float hi) {
    return std::max(lo, std::min(hi, x));
}

// deterministic xorshift32
struct XorShift32 {
    uint32_t s;
    explicit XorShift32(uint32_t seed=0x12345678u) : s(seed ? seed : 0x12345678u) {}
    uint32_t next_u32() {
        uint32_t x = s;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        s = x;
        return x;
    }
    float next_f01() { // [0,1)
        return (next_u32() >> 8) * (1.0f / 16777216.0f);
    }
    int next_int(int lo, int hi) { // inclusive lo, exclusive hi
        uint32_t r = next_u32();
        return lo + int(r % uint32_t(hi - lo));
    }
    // approx normal via Box-Muller
    float next_norm() {
        float u1 = std::max(1e-7f, next_f01());
        float u2 = next_f01();
        float r = std::sqrt(-2.0f * std::log(u1));
        float t = 2.0f * 3.1415926535f * u2;
        return r * std::cos(t);
    }
};

static inline double now_seconds() {
    using clk = std::chrono::high_resolution_clock;
    auto t = clk::now().time_since_epoch();
    return std::chrono::duration<double>(t).count();
}

static void shuffle_inplace(std::vector<int>& v, XorShift32& rng) {
    for (int i = int(v.size()) - 1; i > 0; --i) {
        int j = rng.next_int(0, i + 1);
        std::swap(v[i], v[j]);
    }
}

static float l2_vec(const std::vector<float>& x, const std::vector<float>& y) {
    float s=0;
    for(size_t i=0;i<x.size();++i){
        float d = x[i]-y[i];
        s += d*d;
    }
    return std::sqrt(s);
}

static uint64_t file_size_bytes(const std::string& path){
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if(!f) return 0;
    return (uint64_t)f.tellg();
}

// ============================================================
// Lightweight tensor wrappers
// ============================================================
struct Mat {
    int R=0, C=0;
    std::vector<float> a; // row-major
    Mat() {}
    Mat(int r,int c,float v=0):R(r),C(c),a(size_t(r)*size_t(c),v) {}
    float& operator()(int r,int c){ return a[size_t(r)*size_t(C) + size_t(c)]; }
    const float& operator()(int r,int c) const { return a[size_t(r)*size_t(C) + size_t(c)]; }
};

struct Vec {
    int N=0;
    std::vector<float> a;
    Vec() {}
    Vec(int n,float v=0):N(n),a(size_t(n),v) {}
    float& operator()(int i){ return a[size_t(i)]; }
    const float& operator()(int i) const { return a[size_t(i)]; }
};

// flat indexing helpers
static inline size_t idx3(int n,int t,int d,int T,int D){
    return (size_t(n)*size_t(T) + size_t(t))*size_t(D) + size_t(d);
}
static inline size_t idx2(int n,int t,int T){
    return size_t(n)*size_t(T) + size_t(t);
}

// ============================================================
// Dataset generation (MS_all, A_base, window_delta, PN pilot)
// ============================================================
static Mat MS_all;    // [N_IDENTITIES, MS_DIM]
static Mat A_base;    // [SEQ_LEN, MS_DIM]

static std::vector<float> window_delta_vec(int window_global_id, int t) {
    uint32_t seed = uint32_t((window_global_id * 10007 + t * 97) & 0xFFFFFFFFu);
    XorShift32 rng(seed ? seed : 0xA5A5A5A5u);
    std::vector<float> d(MS_DIM);
    for(int i=0;i<MS_DIM;i++){
        d[i] = 0.25f * rng.next_norm();
    }
    return d;
}

static std::vector<float> window_pilot_vec(int window_global_id) {
    uint32_t seed = uint32_t((window_global_id * 9176 + 11) & 0xFFFFFFFFu);
    XorShift32 rng(seed ? seed : 0xBEEFBEEFu);
    std::vector<float> pilot(SEQ_LEN, 0.0f);
    float mean=0.0f;
    for(int t=0;t<SEQ_LEN;t++){
        int bit = rng.next_int(0,2);
        float chip = (bit==0) ? -1.0f : 1.0f;
        pilot[t] = PILOT_AMP * chip;
        mean += pilot[t];
    }
    mean /= float(SEQ_LEN);
    for(int t=0;t<SEQ_LEN;t++) pilot[t] -= mean;
    return pilot;
}

static float pilot_corr(const std::vector<float>& meas, int window_global_id) {
    const std::vector<float> pilot = window_pilot_vec(window_global_id);
    float num=0.0f, a2=0.0f, b2=0.0f;
    for(int t=0;t<SEQ_LEN;t++){
        num += meas[size_t(t)] * pilot[size_t(t)];
        a2  += meas[size_t(t)] * meas[size_t(t)];
        b2  += pilot[size_t(t)] * pilot[size_t(t)];
    }
    return num / (std::sqrt(a2*b2) + 1e-9f);
}

// Generate OS chain -> tokens + normalized measurements m[t]
static void generate_os_chain(
    const std::vector<float>& ms_vec,
    int window_global_id,
    std::vector<int>& tokens_out,
    std::vector<float>& meas_out
) {
    std::vector<float> zs(SEQ_LEN, 0.0f);

    for(int t=0;t<SEQ_LEN;t++){
        const std::vector<float> d = window_delta_vec(window_global_id, t);
        float dot=0.0f;
        for(int k=0;k<MS_DIM;k++){
            float a = A_base(t,k) + d[size_t(k)];
            dot += a * ms_vec[size_t(k)];
        }
        zs[size_t(t)] = dot;
    }

    // add PN pilot watermark
    const std::vector<float> pilot = window_pilot_vec(window_global_id);
    for(int t=0;t<SEQ_LEN;t++) zs[size_t(t)] += pilot[size_t(t)];

    // small noise (seeded)
    int ms_sum = 0;
    for(int k=0;k<MS_DIM;k++) ms_sum += int(ms_vec[size_t(k)]*1000.0f);
    uint32_t nseed = uint32_t((window_global_id * 1337 + ms_sum) & 0xFFFFFFFFu);
    XorShift32 rng(nseed ? nseed : 0xCAFE1234u);
    for(int t=0;t<SEQ_LEN;t++){
        zs[size_t(t)] += 0.02f * rng.next_norm();
    }

    // normalize zs -> m
    float mean=0.0f;
    for(float v: zs) mean += v;
    mean /= float(SEQ_LEN);

    float var=0.0f;
    for(float v: zs){
        float d = v - mean;
        var += d*d;
    }
    var /= float(SEQ_LEN);
    float st = std::sqrt(var) + 1e-6f;

    meas_out.assign(SEQ_LEN, 0.0f);
    for(int t=0;t<SEQ_LEN;t++){
        meas_out[size_t(t)] = (zs[size_t(t)] - mean) / st;
    }

    // quantize
    tokens_out.assign(SEQ_LEN, 0);
    for(int t=0;t<SEQ_LEN;t++){
        float scaled = clampf((meas_out[size_t(t)] + 3.0f) / 6.0f, 0.0f, 0.999999f);
        int tok = int(scaled * float(VOCAB_SIZE));
        if(tok < 0) tok = 0;
        if(tok >= VOCAB_SIZE) tok = VOCAB_SIZE-1;
        tokens_out[size_t(t)] = tok;
    }
}

static void build_X_backbone(
    const std::vector<int>& tokens,
    const std::vector<float>& meas,
    Mat& X_out  // [SEQ_LEN, INPUT_DIM]
) {
    X_out = Mat(SEQ_LEN, INPUT_DIM, 0.0f);
    for(int t=0;t<SEQ_LEN;t++){
        int tok = tokens[size_t(t)];
        X_out(t, tok) = 1.0f;
        X_out(t, VOCAB_SIZE) = meas[size_t(t)];
        X_out(t, VOCAB_SIZE + 1) = (SEQ_LEN <= 1) ? 0.0f : (float(t) / float(SEQ_LEN-1));
    }
}

struct Dataset {
    int N=0;
    std::vector<float> X;     // N*T*D
    std::vector<float> M;     // N*T
    std::vector<int>   TOK;   // N*T
    std::vector<float> Y_MS;  // N*MS_DIM
    std::vector<float> Y_CLS; // N
    std::vector<int> TRUE_ID, TRUE_W, CLAIM_ID, EXPECT_W;
};

static Dataset build_dataset() {
    Dataset ds;

    const int samples_per_pair = 5;
    ds.N = N_IDENTITIES * N_WINDOWS_PER_ID * samples_per_pair;

    ds.X.assign(size_t(ds.N)*SEQ_LEN*INPUT_DIM, 0.0f);
    ds.M.assign(size_t(ds.N)*SEQ_LEN, 0.0f);
    ds.TOK.assign(size_t(ds.N)*SEQ_LEN, 0);
    ds.Y_MS.assign(size_t(ds.N)*MS_DIM, 0.0f);
    ds.Y_CLS.assign(size_t(ds.N), 0.0f);

    ds.TRUE_ID.assign(ds.N, 0);
    ds.TRUE_W.assign(ds.N, 0);
    ds.CLAIM_ID.assign(ds.N, 0);
    ds.EXPECT_W.assign(ds.N, 0);

    XorShift32 rng(0xBADC0DEu);

    int n=0;
    for(int id_true=0; id_true<N_IDENTITIES; id_true++){
        std::vector<float> ms_true(MS_DIM);
        for(int k=0;k<MS_DIM;k++) ms_true[size_t(k)] = MS_all(id_true,k);

        for(int w_true=0; w_true<N_WINDOWS_PER_ID; w_true++){
            int g_true = id_true * N_WINDOWS_PER_ID + w_true;

            std::vector<int> toks;
            std::vector<float> meas;
            generate_os_chain(ms_true, g_true, toks, meas);

            auto push_sample = [&](const std::vector<int>& toks_in,
                                   const std::vector<float>& meas_in,
                                   float ycls,
                                   int true_id, int true_w,
                                   int claim_id, int expect_w,
                                   const std::vector<float>& yms){
                Mat X;
                build_X_backbone(toks_in, meas_in, X);
                for(int t=0;t<SEQ_LEN;t++){
                    ds.M[idx2(n,t,SEQ_LEN)] = 1.0f;
                    ds.TOK[idx2(n,t,SEQ_LEN)] = toks_in[size_t(t)];
                    for(int d=0; d<INPUT_DIM; d++){
                        ds.X[idx3(n,t,d,SEQ_LEN,INPUT_DIM)] = X(t,d);
                    }
                }
                for(int k=0;k<MS_DIM;k++){
                    ds.Y_MS[size_t(n)*MS_DIM + size_t(k)] = yms[size_t(k)];
                }
                ds.Y_CLS[size_t(n)] = ycls;

                ds.TRUE_ID[n] = true_id;
                ds.TRUE_W[n] = true_w;
                ds.CLAIM_ID[n] = claim_id;
                ds.EXPECT_W[n] = expect_w;

                n++;
            };

            // POS legit
            push_sample(toks, meas, 1.0f, id_true, w_true, id_true, w_true, ms_true);

            // NEG shuffled
            {
                std::vector<int> idxs(SEQ_LEN);
                std::iota(idxs.begin(), idxs.end(), 0);
                XorShift32 rsh(uint32_t(0x1234u + uint32_t(g_true)));
                shuffle_inplace(idxs, rsh);
                std::vector<int> toks2(SEQ_LEN);
                std::vector<float> meas2(SEQ_LEN);
                for(int t=0;t<SEQ_LEN;t++){ toks2[size_t(t)]=toks[size_t(idxs[size_t(t)])]; meas2[size_t(t)]=meas[size_t(idxs[size_t(t)])]; }
                push_sample(toks2, meas2, 0.0f, id_true, w_true, id_true, w_true, ms_true);
            }

            // NEG truncated (half filled, rest masked)
            {
                int Ltr = SEQ_LEN/2;
                std::vector<int> toks2(SEQ_LEN, 0);
                std::vector<float> meas2(SEQ_LEN, 0.0f);
                for(int t=0;t<Ltr;t++){ toks2[size_t(t)]=toks[size_t(t)]; meas2[size_t(t)]=meas[size_t(t)]; }

                Mat X;
                build_X_backbone(toks2, meas2, X);
                int nn = n;
                for(int t=0;t<SEQ_LEN;t++){
                    ds.M[idx2(nn,t,SEQ_LEN)] = (t < Ltr) ? 1.0f : 0.0f;
                    ds.TOK[idx2(nn,t,SEQ_LEN)] = toks2[size_t(t)];
                    for(int d=0; d<INPUT_DIM; d++){
                        ds.X[idx3(nn,t,d,SEQ_LEN,INPUT_DIM)] = X(t,d);
                    }
                }
                for(int k=0;k<MS_DIM;k++) ds.Y_MS[size_t(nn)*MS_DIM + size_t(k)] = ms_true[size_t(k)];
                ds.Y_CLS[size_t(nn)] = 0.0f;
                ds.TRUE_ID[nn]=id_true; ds.TRUE_W[nn]=w_true;
                ds.CLAIM_ID[nn]=id_true; ds.EXPECT_W[nn]=w_true;
                n++;
            }

            // NEG wrong-window chain (claim expects w_true)
            {
                int wrong_w = (w_true + 7) % N_WINDOWS_PER_ID;
                int g_wrong = id_true * N_WINDOWS_PER_ID + wrong_w;
                std::vector<int> toks_w;
                std::vector<float> meas_w;
                generate_os_chain(ms_true, g_wrong, toks_w, meas_w);
                push_sample(toks_w, meas_w, 0.0f, id_true, wrong_w, id_true, w_true, ms_true);
            }

            // NEG wrong identity chain (claim id_true)
            {
                int other_id = (id_true + rng.next_int(1, N_IDENTITIES)) % N_IDENTITIES;
                int other_w  = rng.next_int(0, N_WINDOWS_PER_ID);
                int g_other  = other_id * N_WINDOWS_PER_ID + other_w;
                std::vector<float> ms_other(MS_DIM);
                for(int k=0;k<MS_DIM;k++) ms_other[size_t(k)] = MS_all(other_id,k);
                std::vector<int> toks_o;
                std::vector<float> meas_o;
                generate_os_chain(ms_other, g_other, toks_o, meas_o);
                push_sample(toks_o, meas_o, 0.0f, other_id, other_w, id_true, w_true, ms_true);
            }
        }
    }

    return ds;
}

// ============================================================
// Model params
// ============================================================
struct Params {
    Mat W_z, U_z; Vec b_z;
    Mat W_r, U_r; Vec b_r;
    Mat W_h, U_h; Vec b_h;

    Mat W_att; Vec v_att;

    Mat W_ms1; Vec b_ms1;
    Mat W_ms2; Vec b_ms2;

    Mat W_tok; Vec b_tok;

    Mat W_id; Vec b_id;
    Mat W_w;  Vec b_w;

    Mat W_beh; Vec b_beh; // [1, HIDDEN+4]

    size_t param_count() const {
        auto cm = [](const Mat& m){ return size_t(m.R)*size_t(m.C); };
        auto cv = [](const Vec& v){ return size_t(v.N); };
        size_t c=0;
        c += cm(W_z)+cm(U_z)+cv(b_z);
        c += cm(W_r)+cm(U_r)+cv(b_r);
        c += cm(W_h)+cm(U_h)+cv(b_h);
        c += cm(W_att)+cv(v_att);
        c += cm(W_ms1)+cv(b_ms1);
        c += cm(W_ms2)+cv(b_ms2);
        c += cm(W_tok)+cv(b_tok);
        c += cm(W_id)+cv(b_id);
        c += cm(W_w)+cv(b_w);
        c += cm(W_beh)+cv(b_beh);
        return c;
    }
    size_t bytes() const { return param_count() * sizeof(float); }
};

static Params init_model(int input_dim, uint32_t seed=0xC0FFEEu) {
    Params p;
    XorShift32 rng(seed);

    auto init_mat = [&](int r,int c,float s){
        Mat m(r,c,0.0f);
        for(size_t i=0;i<m.a.size();++i) m.a[i] = s * rng.next_norm();
        return m;
    };
    auto init_vec0 = [&](int n){
        return Vec(n, 0.0f);
    };

    float s = 0.08f;

    p.W_z = init_mat(HIDDEN_DIM, input_dim, s);
    p.U_z = init_mat(HIDDEN_DIM, HIDDEN_DIM, s);
    p.b_z = init_vec0(HIDDEN_DIM);

    p.W_r = init_mat(HIDDEN_DIM, input_dim, s);
    p.U_r = init_mat(HIDDEN_DIM, HIDDEN_DIM, s);
    p.b_r = init_vec0(HIDDEN_DIM);

    p.W_h = init_mat(HIDDEN_DIM, input_dim, s);
    p.U_h = init_mat(HIDDEN_DIM, HIDDEN_DIM, s);
    p.b_h = init_vec0(HIDDEN_DIM);

    p.W_att = init_mat(ATTN_DIM, HIDDEN_DIM, s);
    p.v_att = Vec(ATTN_DIM, 0.0f);
    for(size_t i=0;i<p.v_att.a.size();++i) p.v_att.a[i] = s * rng.next_norm();

    p.W_ms1 = init_mat(MS_HID, HIDDEN_DIM, s);
    p.b_ms1 = init_vec0(MS_HID);
    p.W_ms2 = init_mat(MS_DIM, MS_HID, s);
    p.b_ms2 = init_vec0(MS_DIM);

    p.W_tok = init_mat(VOCAB_SIZE, HIDDEN_DIM, s);
    p.b_tok = init_vec0(VOCAB_SIZE);

    p.W_id = init_mat(N_IDENTITIES, HIDDEN_DIM, s);
    p.b_id = init_vec0(N_IDENTITIES);

    p.W_w  = init_mat(N_WINDOWS_PER_ID, 3*HIDDEN_DIM, s);
    p.b_w  = init_vec0(N_WINDOWS_PER_ID);

    p.W_beh = init_mat(1, HIDDEN_DIM+4, s);
    p.b_beh = init_vec0(1);

    return p;
}

// ============================================================
// Adam optimizer (per-parameter buffers)
// ============================================================
struct AdamState {
    Params m, v;
    int t=0;
    float lr=0.001f;
    float b1=0.9f, b2=0.999f, eps=1e-8f;

    static Params zeros_like(const Params& p) {
        Params z;
        auto zmat = [](const Mat& m){ return Mat(m.R,m.C,0.0f); };
        auto zvec = [](const Vec& v){ return Vec(v.N,0.0f); };

        z.W_z=zmat(p.W_z); z.U_z=zmat(p.U_z); z.b_z=zvec(p.b_z);
        z.W_r=zmat(p.W_r); z.U_r=zmat(p.U_r); z.b_r=zvec(p.b_r);
        z.W_h=zmat(p.W_h); z.U_h=zmat(p.U_h); z.b_h=zvec(p.b_h);
        z.W_att=zmat(p.W_att); z.v_att=zvec(p.v_att);
        z.W_ms1=zmat(p.W_ms1); z.b_ms1=zvec(p.b_ms1);
        z.W_ms2=zmat(p.W_ms2); z.b_ms2=zvec(p.b_ms2);
        z.W_tok=zmat(p.W_tok); z.b_tok=zvec(p.b_tok);
        z.W_id=zmat(p.W_id); z.b_id=zvec(p.b_id);
        z.W_w=zmat(p.W_w); z.b_w=zvec(p.b_w);
        z.W_beh=zmat(p.W_beh); z.b_beh=zvec(p.b_beh);
        return z;
    }

    explicit AdamState(const Params& p, float lr_) : m(zeros_like(p)), v(zeros_like(p)), lr(lr_) {}

    static inline void step_mat(Mat& w, const Mat& g, Mat& m, Mat& v,
                                int t, float lr, float b1, float b2, float eps, float wd, bool is_bias) {
        const float b1t = std::pow(b1, float(t));
        const float b2t = std::pow(b2, float(t));
        for(size_t i=0;i<w.a.size();++i){
            float grad = g.a[i];
            if(wd > 0.0f && !is_bias) grad += wd * w.a[i];

            m.a[i] = b1*m.a[i] + (1-b1)*grad;
            v.a[i] = b2*v.a[i] + (1-b2)*grad*grad;

            float mhat = m.a[i] / (1 - b1t);
            float vhat = v.a[i] / (1 - b2t);
            w.a[i] -= lr * mhat / (std::sqrt(vhat) + eps);
        }
    }

    static inline void step_vec(Vec& w, const Vec& g, Vec& m, Vec& v,
                                int t, float lr, float b1, float b2, float eps, float wd, bool is_bias) {
        const float b1t = std::pow(b1, float(t));
        const float b2t = std::pow(b2, float(t));
        for(size_t i=0;i<w.a.size();++i){
            float grad = g.a[i];
            if(wd > 0.0f && !is_bias) grad += wd * w.a[i];

            m.a[i] = b1*m.a[i] + (1-b1)*grad;
            v.a[i] = b2*v.a[i] + (1-b2)*grad*grad;

            float mhat = m.a[i] / (1 - b1t);
            float vhat = v.a[i] / (1 - b2t);
            w.a[i] -= lr * mhat / (std::sqrt(vhat) + eps);
        }
    }

    struct Freeze {
        bool W_z=false,U_z=false,b_z=false;
        bool W_r=false,U_r=false,b_r=false;
        bool W_h=false,U_h=false,b_h=false;
        bool W_att=false,v_att=false;
        bool W_ms1=false,b_ms1=false,W_ms2=false,b_ms2=false;
        bool W_tok=false,b_tok=false;
        bool W_id=false,b_id=false;
        bool W_w=false,b_w=false;
        bool W_beh=false,b_beh=false;
    };

    void step(Params& p, const Params& g, float weight_decay, const Freeze& fr) {
        t += 1;

        auto SM = [&](Mat& W, const Mat& dW, Mat& mW, Mat& vW, bool freeze, bool is_bias=false){
            if(freeze) return;
            step_mat(W, dW, mW, vW, t, lr, b1, b2, eps, weight_decay, is_bias);
        };
        auto SV = [&](Vec& b, const Vec& db, Vec& mb, Vec& vb, bool freeze, bool is_bias=true){
            if(freeze) return;
            step_vec(b, db, mb, vb, t, lr, b1, b2, eps, weight_decay, is_bias);
        };

        SM(p.W_z, g.W_z, m.W_z, v.W_z, fr.W_z, false);  SM(p.U_z, g.U_z, m.U_z, v.U_z, fr.U_z, false);  SV(p.b_z, g.b_z, m.b_z, v.b_z, fr.b_z, true);
        SM(p.W_r, g.W_r, m.W_r, v.W_r, fr.W_r, false);  SM(p.U_r, g.U_r, m.U_r, v.U_r, fr.U_r, false);  SV(p.b_r, g.b_r, m.b_r, v.b_r, fr.b_r, true);
        SM(p.W_h, g.W_h, m.W_h, v.W_h, fr.W_h, false);  SM(p.U_h, g.U_h, m.U_h, v.U_h, fr.U_h, false);  SV(p.b_h, g.b_h, m.b_h, v.b_h, fr.b_h, true);

        SM(p.W_att, g.W_att, m.W_att, v.W_att, fr.W_att, false); SV(p.v_att, g.v_att, m.v_att, v.v_att, fr.v_att, true);

        SM(p.W_ms1, g.W_ms1, m.W_ms1, v.W_ms1, fr.W_ms1, false); SV(p.b_ms1, g.b_ms1, m.b_ms1, v.b_ms1, fr.b_ms1, true);
        SM(p.W_ms2, g.W_ms2, m.W_ms2, v.W_ms2, fr.W_ms2, false); SV(p.b_ms2, g.b_ms2, m.b_ms2, v.b_ms2, fr.b_ms2, true);

        SM(p.W_tok, g.W_tok, m.W_tok, v.W_tok, fr.W_tok, false); SV(p.b_tok, g.b_tok, m.b_tok, v.b_tok, fr.b_tok, true);

        SM(p.W_id, g.W_id, m.W_id, v.W_id, fr.W_id, false); SV(p.b_id, g.b_id, m.b_id, v.b_id, fr.b_id, true);
        SM(p.W_w,  g.W_w,  m.W_w,  v.W_w,  fr.W_w,  false); SV(p.b_w,  g.b_w,  m.b_w,  v.b_w,  fr.b_w,  true);

        SM(p.W_beh, g.W_beh, m.W_beh, v.W_beh, fr.W_beh, false); SV(p.b_beh, g.b_beh, m.b_beh, v.b_beh, fr.b_beh, true);
    }
};

// ============================================================
// Softmax helpers
// ============================================================
static void softmax1d(const std::vector<float>& logits, std::vector<float>& probs) {
    float mx = -1e30f;
    for(float v: logits) mx = std::max(mx, v);
    float s=0.0f;
    probs.resize(logits.size());
    for(size_t i=0;i<logits.size();++i){
        float e = std::exp(logits[i]-mx);
        probs[i]=e; s+=e;
    }
    float inv = 1.0f/(s+1e-12f);
    for(float& p: probs) p*=inv;
}

// masked softmax scores -> alpha
static void softmax_masked(const std::vector<float>& scores, const std::vector<float>& mask,
                           int B, int T, std::vector<float>& alphas) {
    alphas.assign(size_t(B)*size_t(T), 0.0f);
    for(int b=0;b<B;b++){
        float mx = -1e30f;
        for(int t=0;t<T;t++){
            float m = mask[idx2(b,t,T)];
            if(m>0.0f) mx = std::max(mx, scores[idx2(b,t,T)]);
        }
        float s=0.0f;
        for(int t=0;t<T;t++){
            float m = mask[idx2(b,t,T)];
            if(m<=0.0f) continue;
            float e = std::exp(scores[idx2(b,t,T)] - mx);
            alphas[idx2(b,t,T)] = e;
            s += e;
        }
        float inv = 1.0f/(s+1e-12f);
        for(int t=0;t<T;t++){
            alphas[idx2(b,t,T)] *= inv;
        }
    }
}

// ============================================================
// Forward: GRU (batch) + attention + MS head
// ============================================================
struct GRUCache {
    int B=0, T=0, D=0;
    std::vector<float> X;   // B*T*D
    std::vector<float> M;   // B*T
    std::vector<float> H;   // B*T*HIDDEN
    std::vector<float> Z;   // B*T*HIDDEN
    std::vector<float> R;   // B*T*HIDDEN
    std::vector<float> HT;  // B*T*HIDDEN (htilde)
};

static void gru_forward_batch(
    const Params& p,
    const std::vector<float>& Xb, // B*T*D
    const std::vector<float>& Mb, // B*T
    int B, int T, int D,
    std::vector<float>& H_out, // B*T*HIDDEN
    GRUCache& cache
) {
    H_out.assign(size_t(B)*size_t(T)*size_t(HIDDEN_DIM), 0.0f);
    cache = GRUCache();
    cache.B=B; cache.T=T; cache.D=D;
    cache.X = Xb;
    cache.M = Mb;
    cache.H.assign(size_t(B)*size_t(T)*size_t(HIDDEN_DIM), 0.0f);
    cache.Z.assign(size_t(B)*size_t(T)*size_t(HIDDEN_DIM), 0.0f);
    cache.R.assign(size_t(B)*size_t(T)*size_t(HIDDEN_DIM), 0.0f);
    cache.HT.assign(size_t(B)*size_t(T)*size_t(HIDDEN_DIM), 0.0f);

    std::vector<float> h_prev(size_t(B)*size_t(HIDDEN_DIM), 0.0f);

    for(int t=0;t<T;t++){
        for(int b=0;b<B;b++){
            float mt = Mb[idx2(b,t,T)];

            for(int h=0; h<HIDDEN_DIM; h++){
                float az = p.b_z(h);
                float ar = p.b_r(h);
                float ah = p.b_h(h);

                const float* xptr = &Xb[idx3(b,t,0,T,D)];
                for(int d=0; d<D; d++){
                    float x = xptr[d];
                    az += p.W_z(h,d) * x;
                    ar += p.W_r(h,d) * x;
                    ah += p.W_h(h,d) * x;
                }

                const float* hp = &h_prev[size_t(b)*size_t(HIDDEN_DIM)];
                float uz=0.0f, ur=0.0f, uh=0.0f;
                for(int k=0;k<HIDDEN_DIM;k++){
                    float hv = hp[k];
                    uz += p.U_z(h,k) * hv;
                    ur += p.U_r(h,k) * hv;
                }

                float z = sigmoid(az + uz);
                float r = sigmoid(ar + ur);

                for(int k=0;k<HIDDEN_DIM;k++){
                    uh += p.U_h(h,k) * (r * hp[k]);
                }

                float htil = fast_tanh(ah + uh);
                float hnew = (1.0f - z) * hp[h] + z * htil;
                hnew = mt * hnew + (1.0f - mt) * hp[h];

                H_out[idx3(b,t,h,T,HIDDEN_DIM)] = hnew;
                cache.H[idx3(b,t,h,T,HIDDEN_DIM)] = hnew;
                cache.Z[idx3(b,t,h,T,HIDDEN_DIM)] = z;
                cache.R[idx3(b,t,h,T,HIDDEN_DIM)] = r;
                cache.HT[idx3(b,t,h,T,HIDDEN_DIM)] = htil;
            }

            for(int h=0; h<HIDDEN_DIM; h++){
                h_prev[size_t(b)*size_t(HIDDEN_DIM) + size_t(h)] = H_out[idx3(b,t,h,T,HIDDEN_DIM)];
            }
        }
    }
}

struct AttnCache {
    int B=0,T=0;
    std::vector<float> H;       // B*T*HIDDEN
    std::vector<float> M;       // B*T
    std::vector<float> U;       // B*T*ATTN_DIM
    std::vector<float> SCORES;  // B*T
    std::vector<float> ALPHA;   // B*T
};

static void attention_forward_batch(
    const Params& p,
    const std::vector<float>& H,  // B*T*HIDDEN
    const std::vector<float>& M,  // B*T
    int B,int T,
    std::vector<float>& ctx_out,  // B*HIDDEN
    AttnCache& cache
) {
    cache = AttnCache();
    cache.B=B; cache.T=T;
    cache.H = H;
    cache.M = M;
    cache.U.assign(size_t(B)*size_t(T)*size_t(ATTN_DIM), 0.0f);
    cache.SCORES.assign(size_t(B)*size_t(T), 0.0f);

    for(int b=0;b<B;b++){
        for(int t=0;t<T;t++){
            for(int a=0;a<ATTN_DIM;a++){
                float s = 0.0f;
                for(int h=0;h<HIDDEN_DIM;h++){
                    s += p.W_att(a,h) * H[idx3(b,t,h,T,HIDDEN_DIM)];
                }
                cache.U[idx3(b,t,a,T,ATTN_DIM)] = fast_tanh(s);
            }
            float sc=0.0f;
            for(int a=0;a<ATTN_DIM;a++){
                sc += p.v_att(a) * cache.U[idx3(b,t,a,T,ATTN_DIM)];
            }
            cache.SCORES[idx2(b,t,T)] = sc;
        }
    }

    softmax_masked(cache.SCORES, cache.M, B, T, cache.ALPHA);

    ctx_out.assign(size_t(B)*size_t(HIDDEN_DIM), 0.0f);
    for(int b=0;b<B;b++){
        for(int t=0;t<T;t++){
            float a = cache.ALPHA[idx2(b,t,T)];
            for(int h=0;h<HIDDEN_DIM;h++){
                ctx_out[size_t(b)*size_t(HIDDEN_DIM) + size_t(h)] += a * H[idx3(b,t,h,T,HIDDEN_DIM)];
            }
        }
    }
}

static void ms_head_forward(
    const Params& p,
    const std::vector<float>& ctx, // B*HIDDEN
    int B,
    std::vector<float>& ms_hat,    // B*MS_DIM
    std::vector<float>& ms_hid     // B*MS_HID (tanh output)
) {
    ms_hid.assign(size_t(B)*size_t(MS_HID), 0.0f);
    ms_hat.assign(size_t(B)*size_t(MS_DIM), 0.0f);

    for(int b=0;b<B;b++){
        for(int j=0;j<MS_HID;j++){
            float s = p.b_ms1(j);
            for(int h=0;h<HIDDEN_DIM;h++){
                s += p.W_ms1(j,h) * ctx[size_t(b)*size_t(HIDDEN_DIM) + size_t(h)];
            }
            ms_hid[size_t(b)*size_t(MS_HID) + size_t(j)] = fast_tanh(s);
        }
        for(int k=0;k<MS_DIM;k++){
            float s = p.b_ms2(k);
            for(int j=0;j<MS_HID;j++){
                s += p.W_ms2(k,j) * ms_hid[size_t(b)*size_t(MS_HID) + size_t(j)];
            }
            ms_hat[size_t(b)*size_t(MS_DIM) + size_t(k)] = s;
        }
    }
}

// ============================================================
// Grad helpers
// ============================================================
static Params grads_zeros_like(const Params& p) {
    return AdamState::zeros_like(p);
}

static float global_norm(const Params& g) {
    auto sumsq_mat=[&](const Mat& m){
        double s=0; for(float v: m.a) s += double(v)*double(v); return s;
    };
    auto sumsq_vec=[&](const Vec& v){
        double s=0; for(float x: v.a) s += double(x)*double(x); return s;
    };
    double s=0;
    s += sumsq_mat(g.W_z)+sumsq_mat(g.U_z)+sumsq_vec(g.b_z);
    s += sumsq_mat(g.W_r)+sumsq_mat(g.U_r)+sumsq_vec(g.b_r);
    s += sumsq_mat(g.W_h)+sumsq_mat(g.U_h)+sumsq_vec(g.b_h);
    s += sumsq_mat(g.W_att)+sumsq_vec(g.v_att);
    s += sumsq_mat(g.W_ms1)+sumsq_vec(g.b_ms1)+sumsq_mat(g.W_ms2)+sumsq_vec(g.b_ms2);
    s += sumsq_mat(g.W_tok)+sumsq_vec(g.b_tok);
    s += sumsq_mat(g.W_id)+sumsq_vec(g.b_id);
    s += sumsq_mat(g.W_w)+sumsq_vec(g.b_w);
    s += sumsq_mat(g.W_beh)+sumsq_vec(g.b_beh);
    return float(std::sqrt(s));
}

static void clip_grads(Params& g, float max_norm) {
    float n = global_norm(g);
    if(n <= max_norm) return;
    float s = max_norm / (n + 1e-12f);

    auto scale_mat=[&](Mat& m){ for(float& v: m.a) v*=s; };
    auto scale_vec=[&](Vec& v){ for(float& x: v.a) x*=s; };

    scale_mat(g.W_z); scale_mat(g.U_z); scale_vec(g.b_z);
    scale_mat(g.W_r); scale_mat(g.U_r); scale_vec(g.b_r);
    scale_mat(g.W_h); scale_mat(g.U_h); scale_vec(g.b_h);
    scale_mat(g.W_att); scale_vec(g.v_att);
    scale_mat(g.W_ms1); scale_vec(g.b_ms1); scale_mat(g.W_ms2); scale_vec(g.b_ms2);
    scale_mat(g.W_tok); scale_vec(g.b_tok);
    scale_mat(g.W_id); scale_vec(g.b_id);
    scale_mat(g.W_w);  scale_vec(g.b_w);
    scale_mat(g.W_beh); scale_vec(g.b_beh);
}

// Token CE for one row
static float token_ce_one(const std::vector<float>& logits, int target, std::vector<float>& dlogits) {
    std::vector<float> p;
    softmax1d(logits, p);
    float loss = -std::log(p[size_t(target)] + 1e-12f);
    dlogits = p;
    dlogits[size_t(target)] -= 1.0f;
    return loss;
}

// ============================================================
// Batch extraction
// ============================================================
static void extract_batch(const Dataset& ds, const std::vector<int>& indices, int start, int B,
                          std::vector<float>& Xb, std::vector<float>& Mb, std::vector<int>& Tb,
                          std::vector<float>& yms, std::vector<float>& ycls) {
    int bsz = std::min(B, int(indices.size()) - start);
    Xb.assign(size_t(bsz)*size_t(SEQ_LEN)*size_t(INPUT_DIM), 0.0f);
    Mb.assign(size_t(bsz)*size_t(SEQ_LEN), 0.0f);
    Tb.assign(size_t(bsz)*size_t(SEQ_LEN), 0);
    yms.assign(size_t(bsz)*size_t(MS_DIM), 0.0f);
    ycls.assign(size_t(bsz), 0.0f);

    for(int bi=0; bi<bsz; bi++){
        int n = indices[size_t(start + bi)];
        for(int t=0;t<SEQ_LEN;t++){
            Mb[idx2(bi,t,SEQ_LEN)] = ds.M[idx2(n,t,SEQ_LEN)];
            Tb[idx2(bi,t,SEQ_LEN)] = ds.TOK[idx2(n,t,SEQ_LEN)];
            for(int d=0; d<INPUT_DIM; d++){
                Xb[idx3(bi,t,d,SEQ_LEN,INPUT_DIM)] = ds.X[idx3(n,t,d,SEQ_LEN,INPUT_DIM)];
            }
        }
        for(int k=0;k<MS_DIM;k++){
            yms[size_t(bi)*size_t(MS_DIM) + size_t(k)] = ds.Y_MS[size_t(n)*size_t(MS_DIM) + size_t(k)];
        }
        ycls[size_t(bi)] = ds.Y_CLS[size_t(n)];
    }
}

// ============================================================
// Phase1 training: MS reconstruction + token scaffold
// ============================================================
static void train_phase1(Params& p, const Dataset& ds) {
    AdamState opt(p, LR_PHASE1);
    AdamState::Freeze nofreeze;

    bool tok_enabled = true;
    // tok_enabled = (epoch < TOK_WARMUP_EPOCHS);
    // if(tok_enabled && epoch > TOK_WARMUP_EPOCHS) tok_enabled = false;

    std::vector<int> idx(ds.N);
    std::iota(idx.begin(), idx.end(), 0);

    for(int ep=1; ep<=EPOCHS_PHASE1; ep++){
        XorShift32 sh(uint32_t(0x1000u + uint32_t(ep)));
        shuffle_inplace(idx, sh);

        double epoch_loss_sum = 0.0;

        for(int s=0; s<ds.N; s+=BATCH_SIZE){
            std::vector<float> Xb, Mb, yms, ycls;
            std::vector<int> Tb;
            extract_batch(ds, idx, s, BATCH_SIZE, Xb, Mb, Tb, yms, ycls);
            int B = int(ycls.size());

            std::vector<float> H;
            GRUCache gcache;
            gru_forward_batch(p, Xb, Mb, B, SEQ_LEN, INPUT_DIM, H, gcache);

            std::vector<float> ctx;
            AttnCache acache;
            attention_forward_batch(p, H, Mb, B, SEQ_LEN, ctx, acache);

            std::vector<float> ms_hat, ms_hid;
            ms_head_forward(p, ctx, B, ms_hat, ms_hid);

            float pos_count = 0.0f;
            for(int i=0;i<B;i++) if(ycls[size_t(i)] > 0.5f) pos_count += 1.0f;
            pos_count += 1e-6f;

            float loss_ms = 0.0f;
            std::vector<float> diff(size_t(B)*size_t(MS_DIM), 0.0f);
            for(int i=0;i<B;i++){
                float pm = (ycls[size_t(i)] > 0.5f) ? 1.0f : 0.0f;
                for(int k=0;k<MS_DIM;k++){
                    float d = (ms_hat[size_t(i)*size_t(MS_DIM) + size_t(k)] - yms[size_t(i)*size_t(MS_DIM) + size_t(k)]) * pm;
                    diff[size_t(i)*size_t(MS_DIM) + size_t(k)] = d;
                    loss_ms += 0.5f * d*d;
                }
            }
            loss_ms /= (pos_count * float(MS_DIM));

            float loss_tok = 0.0f;
            std::vector<float> dH_tok(size_t(B)*size_t(SEQ_LEN)*size_t(HIDDEN_DIM), 0.0f);
            Mat dW_tok(p.W_tok.R, p.W_tok.C, 0.0f);
            Vec db_tok(p.b_tok.N, 0.0f);

            if(tok_enabled && ep <= TOK_WARMUP_EPOCHS){
                float denom = 0.0f;
                for(int i=0;i<B;i++){
                    if(ycls[size_t(i)] <= 0.5f) continue;
                    for(int t=0;t<SEQ_LEN-1;t++){
                        float v = Mb[idx2(i,t,SEQ_LEN)] * Mb[idx2(i,t+1,SEQ_LEN)];
                        if(v>0.5f) denom += 1.0f;
                    }
                }
                denom += 1e-6f;

                for(int i=0;i<B;i++){
                    if(ycls[size_t(i)] <= 0.5f) continue;
                    for(int t=0;t<SEQ_LEN-1;t++){
                        float v = Mb[idx2(i,t,SEQ_LEN)] * Mb[idx2(i,t+1,SEQ_LEN)];
                        if(v<=0.5f) continue;

                        std::vector<float> logits(VOCAB_SIZE, 0.0f);
                        for(int vj=0; vj<VOCAB_SIZE; vj++){
                            float s0 = p.b_tok(vj);
                            for(int h=0; h<HIDDEN_DIM; h++){
                                s0 += p.W_tok(vj,h) * H[idx3(i,t,h,SEQ_LEN,HIDDEN_DIM)];
                            }
                            logits[size_t(vj)] = s0;
                        }
                        int target = Tb[idx2(i,t+1,SEQ_LEN)];
                        std::vector<float> dlog;
                        float li = token_ce_one(logits, target, dlog);
                        loss_tok += li;

                        for(int vj=0; vj<VOCAB_SIZE; vj++){
                            db_tok(vj) += dlog[size_t(vj)];
                            for(int h=0; h<HIDDEN_DIM; h++){
                                dW_tok(vj,h) += dlog[size_t(vj)] * H[idx3(i,t,h,SEQ_LEN,HIDDEN_DIM)];
                            }
                        }
                        for(int h=0; h<HIDDEN_DIM; h++){
                            float sH = 0.0f;
                            for(int vj=0; vj<VOCAB_SIZE; vj++){
                                sH += dlog[size_t(vj)] * p.W_tok(vj,h);
                            }
                            dH_tok[idx3(i,t,h,SEQ_LEN,HIDDEN_DIM)] += sH;
                        }
                    }
                }

                loss_tok /= denom;
                for(float& v: dW_tok.a) v /= denom;
                for(float& v: db_tok.a) v /= denom;
                for(float& v: dH_tok)   v /= denom;

                if(loss_tok < TOK_STOP_EPS) tok_enabled = false;
                // bool tok_enabled = (ep <= TOK_WARMUP_EPOCHS);

            } else {
                tok_enabled = false;
            }

            float loss = LAMBDA_MS*loss_ms + (tok_enabled ? (LAMBDA_TOK*loss_tok) : 0.0f);
            epoch_loss_sum += double(loss) * double(B);

            Params grads = grads_zeros_like(p);
            grads.W_tok = dW_tok;
            grads.b_tok = db_tok;

            std::vector<float> dms(size_t(B)*size_t(MS_DIM), 0.0f);
            for(int i=0;i<B;i++){
                for(int k=0;k<MS_DIM;k++){
                    dms[size_t(i)*size_t(MS_DIM) + size_t(k)] =
                        diff[size_t(i)*size_t(MS_DIM) + size_t(k)] / (pos_count * float(MS_DIM));
                }
            }

            for(int k=0;k<MS_DIM;k++){
                float sb=0.0f;
                for(int i=0;i<B;i++) sb += dms[size_t(i)*size_t(MS_DIM) + size_t(k)];
                grads.b_ms2(k) += sb;
                for(int j=0;j<MS_HID;j++){
                    float sW=0.0f;
                    for(int i=0;i<B;i++){
                        sW += dms[size_t(i)*size_t(MS_DIM) + size_t(k)] * ms_hid[size_t(i)*size_t(MS_HID) + size_t(j)];
                    }
                    grads.W_ms2(k,j) += sW;
                }
            }

            std::vector<float> dms_hid(size_t(B)*size_t(MS_HID), 0.0f);
            for(int i=0;i<B;i++){
                for(int j=0;j<MS_HID;j++){
                    float s=0.0f;
                    for(int k=0;k<MS_DIM;k++){
                        s += dms[size_t(i)*size_t(MS_DIM) + size_t(k)] * p.W_ms2(k,j);
                    }
                    dms_hid[size_t(i)*size_t(MS_HID) + size_t(j)] = s;
                }
            }

            std::vector<float> dpre(size_t(B)*size_t(MS_HID), 0.0f);
            for(int i=0;i<B;i++){
                for(int j=0;j<MS_HID;j++){
                    float h = ms_hid[size_t(i)*size_t(MS_HID) + size_t(j)];
                    dpre[size_t(i)*size_t(MS_HID) + size_t(j)] = dms_hid[size_t(i)*size_t(MS_HID) + size_t(j)] * (1.0f - h*h);
                }
            }

            std::vector<float> dctx(size_t(B)*size_t(HIDDEN_DIM), 0.0f);
            for(int j=0;j<MS_HID;j++){
                float sb=0.0f;
                for(int i=0;i<B;i++) sb += dpre[size_t(i)*size_t(MS_HID) + size_t(j)];
                grads.b_ms1(j) += sb;
                for(int h=0;h<HIDDEN_DIM;h++){
                    float sW=0.0f;
                    for(int i=0;i<B;i++){
                        sW += dpre[size_t(i)*size_t(MS_HID) + size_t(j)] * ctx[size_t(i)*size_t(HIDDEN_DIM) + size_t(h)];
                    }
                    grads.W_ms1(j,h) += sW;
                }
            }
            for(int i=0;i<B;i++){
                for(int h=0;h<HIDDEN_DIM;h++){
                    float s=0.0f;
                    for(int j=0;j<MS_HID;j++){
                        s += dpre[size_t(i)*size_t(MS_HID) + size_t(j)] * p.W_ms1(j,h);
                    }
                    dctx[size_t(i)*size_t(HIDDEN_DIM) + size_t(h)] = s;
                }
            }

            std::vector<float> dH(size_t(B)*size_t(SEQ_LEN)*size_t(HIDDEN_DIM), 0.0f);

            for(int i=0;i<B;i++){
                for(int t=0;t<SEQ_LEN;t++){
                    float a = acache.ALPHA[idx2(i,t,SEQ_LEN)];
                    for(int h=0;h<HIDDEN_DIM;h++){
                        dH[idx3(i,t,h,SEQ_LEN,HIDDEN_DIM)] += a * dctx[size_t(i)*size_t(HIDDEN_DIM) + size_t(h)];
                    }
                }
            }

            std::vector<float> d_alpha(size_t(B)*size_t(SEQ_LEN), 0.0f);
            for(int i=0;i<B;i++){
                for(int t=0;t<SEQ_LEN;t++){
                    float sA=0.0f;
                    for(int h=0;h<HIDDEN_DIM;h++){
                        sA += dctx[size_t(i)*size_t(HIDDEN_DIM) + size_t(h)] * H[idx3(i,t,h,SEQ_LEN,HIDDEN_DIM)];
                    }
                    d_alpha[idx2(i,t,SEQ_LEN)] = sA;
                }
            }

            std::vector<float> dscores(size_t(B)*size_t(SEQ_LEN), 0.0f);
            for(int i=0;i<B;i++){
                float sum_term=0.0f;
                for(int t=0;t<SEQ_LEN;t++){
                    sum_term += acache.ALPHA[idx2(i,t,SEQ_LEN)] * d_alpha[idx2(i,t,SEQ_LEN)];
                }
                for(int t=0;t<SEQ_LEN;t++){
                    float m = Mb[idx2(i,t,SEQ_LEN)];
                    float a = acache.ALPHA[idx2(i,t,SEQ_LEN)];
                    dscores[idx2(i,t,SEQ_LEN)] = a * (d_alpha[idx2(i,t,SEQ_LEN)] - sum_term) * m;
                }
            }

            for(int a=0;a<ATTN_DIM;a++){
                float sv=0.0f;
                for(int i=0;i<B;i++){
                    for(int t=0;t<SEQ_LEN;t++){
                        sv += dscores[idx2(i,t,SEQ_LEN)] * acache.U[idx3(i,t,a,SEQ_LEN,ATTN_DIM)];
                    }
                }
                grads.v_att(a) += sv;
            }

            for(int i=0;i<B;i++){
                for(int t=0;t<SEQ_LEN;t++){
                    float dsc = dscores[idx2(i,t,SEQ_LEN)];
                    if(Mb[idx2(i,t,SEQ_LEN)] <= 0.0f) continue;

                    std::vector<float> da(ATTN_DIM, 0.0f);
                    for(int a=0;a<ATTN_DIM;a++){
                        float u = acache.U[idx3(i,t,a,SEQ_LEN,ATTN_DIM)];
                        float du = dsc * p.v_att(a);
                        da[size_t(a)] = du * (1.0f - u*u);
                    }

                    for(int a=0;a<ATTN_DIM;a++){
                        for(int h=0;h<HIDDEN_DIM;h++){
                            grads.W_att(a,h) += da[size_t(a)] * H[idx3(i,t,h,SEQ_LEN,HIDDEN_DIM)];
                        }
                    }
                    for(int h=0;h<HIDDEN_DIM;h++){
                        float sH=0.0f;
                        for(int a=0;a<ATTN_DIM;a++){
                            sH += da[size_t(a)] * p.W_att(a,h);
                        }
                        dH[idx3(i,t,h,SEQ_LEN,HIDDEN_DIM)] += sH;
                    }
                }
            }

            for(size_t ii=0; ii<dH.size(); ii++) dH[ii] += dH_tok[ii];

            std::vector<float> dh_next(size_t(B)*size_t(HIDDEN_DIM), 0.0f);

            for(int t=SEQ_LEN-1; t>=0; t--){
                for(int i=0;i<B;i++){
                    float mt = Mb[idx2(i,t,SEQ_LEN)];
                    const float* hprev = nullptr;
                    std::vector<float> hprev0;
                    if(t==0){
                        hprev0.assign(size_t(HIDDEN_DIM), 0.0f);
                        hprev = hprev0.data();
                    } else {
                        hprev = &gcache.H[idx3(i,t-1,0,SEQ_LEN,HIDDEN_DIM)];
                    }

                    const float* z = &gcache.Z[idx3(i,t,0,SEQ_LEN,HIDDEN_DIM)];
                    const float* r = &gcache.R[idx3(i,t,0,SEQ_LEN,HIDDEN_DIM)];
                    const float* htil = &gcache.HT[idx3(i,t,0,SEQ_LEN,HIDDEN_DIM)];

                    std::vector<float> dh(size_t(HIDDEN_DIM), 0.0f);
                    for(int h=0;h<HIDDEN_DIM;h++){
                        float v = dh_next[size_t(i)*size_t(HIDDEN_DIM) + size_t(h)] + dH[idx3(i,t,h,SEQ_LEN,HIDDEN_DIM)];
                        dh[size_t(h)] = v * mt;
                    }

                    std::vector<float> dh_til(size_t(HIDDEN_DIM), 0.0f);
                    std::vector<float> dz(size_t(HIDDEN_DIM), 0.0f);
                    std::vector<float> dh_prev_acc(size_t(HIDDEN_DIM), 0.0f);

                    for(int h=0;h<HIDDEN_DIM;h++){
                        dh_til[size_t(h)] = dh[size_t(h)] * z[h];
                        dz[size_t(h)]     = dh[size_t(h)] * (htil[h] - hprev[h]);
                        dh_prev_acc[size_t(h)] = dh[size_t(h)] * (1.0f - z[h]);
                    }

                    std::vector<float> da_h(size_t(HIDDEN_DIM), 0.0f);
                    for(int h=0;h<HIDDEN_DIM;h++){
                        da_h[size_t(h)] = dh_til[size_t(h)] * (1.0f - htil[h]*htil[h]);
                        grads.b_h(h) += da_h[size_t(h)];
                    }

                    const float* xptr = &Xb[idx3(i,t,0,SEQ_LEN,INPUT_DIM)];
                    for(int h=0;h<HIDDEN_DIM;h++){
                        for(int d=0; d<INPUT_DIM; d++){
                            grads.W_h(h,d) += da_h[size_t(h)] * xptr[d];
                        }
                    }
                    for(int h=0;h<HIDDEN_DIM;h++){
                        for(int k=0;k<HIDDEN_DIM;k++){
                            grads.U_h(h,k) += da_h[size_t(h)] * (r[h] * hprev[k]);
                        }
                    }

                    std::vector<float> tmpU(size_t(HIDDEN_DIM), 0.0f);
                    for(int k=0;k<HIDDEN_DIM;k++){
                        float sU=0.0f;
                        for(int h=0;h<HIDDEN_DIM;h++){
                            sU += da_h[size_t(h)] * p.U_h(h,k);
                        }
                        tmpU[size_t(k)] = sU;
                        dh_prev_acc[size_t(k)] += sU * r[k];
                    }
                    std::vector<float> dr(size_t(HIDDEN_DIM), 0.0f);
                    for(int k=0;k<HIDDEN_DIM;k++){
                        dr[size_t(k)] = tmpU[size_t(k)] * hprev[k];
                    }

                    std::vector<float> da_r(size_t(HIDDEN_DIM), 0.0f);
                    for(int h=0;h<HIDDEN_DIM;h++){
                        da_r[size_t(h)] = dr[size_t(h)] * r[h] * (1.0f - r[h]);
                        grads.b_r(h) += da_r[size_t(h)];
                    }
                    for(int h=0;h<HIDDEN_DIM;h++){
                        for(int d=0; d<INPUT_DIM; d++){
                            grads.W_r(h,d) += da_r[size_t(h)] * xptr[d];
                        }
                        for(int k=0;k<HIDDEN_DIM;k++){
                            grads.U_r(h,k) += da_r[size_t(h)] * hprev[k];
                        }
                    }
                    for(int k=0;k<HIDDEN_DIM;k++){
                        float sU=0.0f;
                        for(int h=0;h<HIDDEN_DIM;h++){
                            sU += da_r[size_t(h)] * p.U_r(h,k);
                        }
                        dh_prev_acc[size_t(k)] += sU;
                    }

                    std::vector<float> da_z(size_t(HIDDEN_DIM), 0.0f);
                    for(int h=0;h<HIDDEN_DIM;h++){
                        da_z[size_t(h)] = dz[size_t(h)] * z[h] * (1.0f - z[h]);
                        grads.b_z(h) += da_z[size_t(h)];
                    }
                    for(int h=0;h<HIDDEN_DIM;h++){
                        for(int d=0; d<INPUT_DIM; d++){
                            grads.W_z(h,d) += da_z[size_t(h)] * xptr[d];
                        }
                        for(int k=0;k<HIDDEN_DIM;k++){
                            grads.U_z(h,k) += da_z[size_t(h)] * hprev[k];
                        }
                    }
                    for(int k=0;k<HIDDEN_DIM;k++){
                        float sU=0.0f;
                        for(int h=0;h<HIDDEN_DIM;h++){
                            sU += da_z[size_t(h)] * p.U_z(h,k);
                        }
                        dh_prev_acc[size_t(k)] += sU;
                    }

                    for(int h=0;h<HIDDEN_DIM;h++){
                        dh_next[size_t(i)*size_t(HIDDEN_DIM) + size_t(h)] = dh_prev_acc[size_t(h)];
                    }
                }
            }

            clip_grads(grads, CLIP_NORM);
            opt.step(p, grads, WEIGHT_DECAY, nofreeze);
        }

        if(ep==2 || ep % std::max(1, EPOCHS_PHASE1/10) == 0){
            cout << "[Phase1] Epoch " << ep << "/" << EPOCHS_PHASE1
                 << " avg_loss=" << (epoch_loss_sum / double(ds.N))
                 << " tok_enabled=" << (tok_enabled ? "true":"false")
                 << endl;
        }
    }
}

// ============================================================
// Embeddings for Phase2: ctx + h_last + h_mean
// ============================================================
static void compute_embeddings_all(
    const Params& p,
    const Dataset& ds,
    std::vector<float>& ctx_all,
    std::vector<float>& hlast_all,
    std::vector<float>& hmean_all
) {
    ctx_all.assign(size_t(ds.N)*size_t(HIDDEN_DIM), 0.0f);
    hlast_all.assign(size_t(ds.N)*size_t(HIDDEN_DIM), 0.0f);
    hmean_all.assign(size_t(ds.N)*size_t(HIDDEN_DIM), 0.0f);

    std::vector<int> idx(ds.N);
    std::iota(idx.begin(), idx.end(), 0);

    for(int s=0; s<ds.N; s+=BATCH_SIZE){
        std::vector<float> Xb, Mb, yms, ycls;
        std::vector<int> Tb;
        extract_batch(ds, idx, s, BATCH_SIZE, Xb, Mb, Tb, yms, ycls);
        int B = int(ycls.size());

        std::vector<float> H;
        GRUCache gcache;
        gru_forward_batch(p, Xb, Mb, B, SEQ_LEN, INPUT_DIM, H, gcache);

        std::vector<float> ctx;
        AttnCache acache;
        attention_forward_batch(p, H, Mb, B, SEQ_LEN, ctx, acache);

        for(int i=0;i<B;i++){
            float denom = 0.0f;
            for(int t=0;t<SEQ_LEN;t++) denom += Mb[idx2(i,t,SEQ_LEN)];
            denom += 1e-6f;

            for(int h=0;h<HIDDEN_DIM;h++){
                float sum=0.0f;
                for(int t=0;t<SEQ_LEN;t++){
                    float m = Mb[idx2(i,t,SEQ_LEN)];
                    sum += m * H[idx3(i,t,h,SEQ_LEN,HIDDEN_DIM)];
                }
                hmean_all[size_t(s+i)*size_t(HIDDEN_DIM) + size_t(h)] = sum / denom;
            }

            int last = 0;
            for(int t=0;t<SEQ_LEN;t++){
                if(Mb[idx2(i,t,SEQ_LEN)] > 0.5f) last = t;
            }
            for(int h=0;h<HIDDEN_DIM;h++){
                hlast_all[size_t(s+i)*size_t(HIDDEN_DIM) + size_t(h)] = H[idx3(i,last,h,SEQ_LEN,HIDDEN_DIM)];
                ctx_all[size_t(s+i)*size_t(HIDDEN_DIM) + size_t(h)] = ctx[size_t(i)*size_t(HIDDEN_DIM) + size_t(h)];
            }
        }
    }
}

// ============================================================
// Phase2 training (heads only): ID CE on positives, W CE on positives, Validity BCE on all
// ============================================================
static float ce_loss_and_grad(
    const std::vector<float>& logits, int C,
    const std::vector<int>& target, const std::vector<float>& mask01,
    int B,
    std::vector<float>& dlogits
) {
    dlogits.assign(size_t(B)*size_t(C), 0.0f);
    int cnt=0;
    float L=0.0f;

    for(int i=0;i<B;i++){
        if(mask01[size_t(i)] <= 0.5f) continue;
        cnt++;
        // std::vector<float> row(size_t(C));
        std::vector<float> row(static_cast<size_t>(C), 0.0f);

        for(int c=0;c<C;c++) row[size_t(c)] = logits[size_t(i)*size_t(C) + size_t(c)];
        std::vector<float> p;
        softmax1d(row, p);
        L += -std::log(p[size_t(target[size_t(i)])] + 1e-12f);
        for(int c=0;c<C;c++){
            float d = p[size_t(c)];
            if(c == target[size_t(i)]) d -= 1.0f;
            dlogits[size_t(i)*size_t(C) + size_t(c)] = d;
        }
    }
    if(cnt==0) return 0.0f;
    float inv = 1.0f / float(cnt);
    L *= inv;
    for(float& v: dlogits) v *= inv;
    return L;
}

static float bce_loss_and_grad(
    const std::vector<float>& logits,
    const std::vector<float>& y,
    int B,
    std::vector<float>& dlog
) {
    dlog.assign(size_t(B), 0.0f);
    float L=0.0f;
    for(int i=0;i<B;i++){
        float p = sigmoid(logits[size_t(i)]);
        float yi = y[size_t(i)];
        float eps=1e-8f;
        float loss = -(POS_WEIGHT*yi*std::log(p+eps) + (1.0f-yi)*std::log(1.0f-p+eps));
        L += loss;

        float dl = (p - yi);
        if(yi > 0.5f) dl *= POS_WEIGHT;
        dlog[size_t(i)] = dl;
    }
    L /= float(B);
    for(float& v: dlog) v /= float(B);
    return L;
}

static void train_phase2(Params& p, const Dataset& ds) {
    AdamState opt(p, LR_PHASE2_BASE);

    AdamState::Freeze fr;
    fr.W_z=fr.U_z=fr.b_z=true;
    fr.W_r=fr.U_r=fr.b_r=true;
    fr.W_h=fr.U_h=fr.b_h=true;
    fr.W_att=fr.v_att=true;
    fr.W_ms1=fr.b_ms1=fr.W_ms2=fr.b_ms2=true;
    fr.W_tok=fr.b_tok=true;
    fr.W_id=fr.b_id=false;
    fr.W_w=fr.b_w=false;
    fr.W_beh=fr.b_beh=false;

    std::vector<int> idx(ds.N);
    std::iota(idx.begin(), idx.end(), 0);

    // Refresh embeddings each epoch to reduce weird numerical “bursts” on some toolchains
    std::vector<float> ctx_all, hlast_all, hmean_all;

    for(int ep=1; ep<=EPOCHS_PHASE2; ep++){
        compute_embeddings_all(p, ds, ctx_all, hlast_all, hmean_all);

        float lr = LR_PHASE2_BASE * std::pow(0.98f, float(ep)/30.0f);
        opt.lr = lr;

        XorShift32 sh(uint32_t(0x9000u + uint32_t(ep)));
        shuffle_inplace(idx, sh);

        double epoch_loss_sum=0.0;

        for(int s=0; s<ds.N; s+=BATCH_SIZE){
            int B = std::min(BATCH_SIZE, ds.N - s);

            std::vector<float> cb(size_t(B)*size_t(HIDDEN_DIM),0.0f);
            std::vector<float> hl(size_t(B)*size_t(HIDDEN_DIM),0.0f);
            std::vector<float> hm(size_t(B)*size_t(HIDDEN_DIM),0.0f);

            std::vector<float> yb(size_t(B),0.0f);
            std::vector<float> pos(size_t(B),0.0f);
            std::vector<int> tid(size_t(B),0), tw(size_t(B),0), claim(size_t(B),0), expw(size_t(B),0);

            for(int i=0;i<B;i++){
                int n = idx[size_t(s+i)];
                for(int h=0;h<HIDDEN_DIM;h++){
                    cb[size_t(i)*size_t(HIDDEN_DIM)+size_t(h)] = ctx_all[size_t(n)*size_t(HIDDEN_DIM)+size_t(h)];
                    hl[size_t(i)*size_t(HIDDEN_DIM)+size_t(h)] = hlast_all[size_t(n)*size_t(HIDDEN_DIM)+size_t(h)];
                    hm[size_t(i)*size_t(HIDDEN_DIM)+size_t(h)] = hmean_all[size_t(n)*size_t(HIDDEN_DIM)+size_t(h)];
                }
                yb[size_t(i)] = ds.Y_CLS[size_t(n)];
                pos[size_t(i)] = (yb[size_t(i)]>0.5f) ? 1.0f : 0.0f;
                tid[size_t(i)] = ds.TRUE_ID[n];
                tw[size_t(i)]  = ds.TRUE_W[n];
                claim[size_t(i)] = ds.CLAIM_ID[n];
                expw[size_t(i)]  = ds.EXPECT_W[n];
            }

            std::vector<float> logits_id(size_t(B)*size_t(N_IDENTITIES), 0.0f);
            for(int i=0;i<B;i++){
                for(int c=0;c<N_IDENTITIES;c++){
                    float s0 = p.b_id(c);
                    for(int h=0;h<HIDDEN_DIM;h++){
                        s0 += p.W_id(c,h) * cb[size_t(i)*size_t(HIDDEN_DIM) + size_t(h)];
                    }
                    logits_id[size_t(i)*size_t(N_IDENTITIES) + size_t(c)] = s0;
                }
            }

            std::vector<float> feat_w(size_t(B)*size_t(3*HIDDEN_DIM), 0.0f);
            for(int i=0;i<B;i++){
                for(int h=0;h<HIDDEN_DIM;h++){
                    feat_w[size_t(i)*size_t(3*HIDDEN_DIM) + size_t(h)] = cb[size_t(i)*size_t(HIDDEN_DIM)+size_t(h)];
                    feat_w[size_t(i)*size_t(3*HIDDEN_DIM) + size_t(HIDDEN_DIM+h)] = hl[size_t(i)*size_t(HIDDEN_DIM)+size_t(h)];
                    feat_w[size_t(i)*size_t(3*HIDDEN_DIM) + size_t(2*HIDDEN_DIM+h)] = hm[size_t(i)*size_t(HIDDEN_DIM)+size_t(h)];
                }
            }

            std::vector<float> logits_w(size_t(B)*size_t(N_WINDOWS_PER_ID), 0.0f);
            for(int i=0;i<B;i++){
                for(int c=0;c<N_WINDOWS_PER_ID;c++){
                    float s0 = p.b_w(c);
                    for(int j=0;j<3*HIDDEN_DIM;j++){
                        s0 += p.W_w(c,j) * feat_w[size_t(i)*size_t(3*HIDDEN_DIM) + size_t(j)];
                    }
                    logits_w[size_t(i)*size_t(N_WINDOWS_PER_ID) + size_t(c)] = s0;
                }
            }

            std::vector<float> p_id_claimed(size_t(B),0.0f);
            std::vector<float> p_w_expected(size_t(B),0.0f);

            for(int i=0;i<B;i++){
                // std::vector<float> row_id(size_t(N_IDENTITIES));
                std::vector<float> row_id(static_cast<size_t>(N_IDENTITIES), 0.0f);
                for(int c=0;c<N_IDENTITIES;c++) row_id[size_t(c)] = logits_id[size_t(i)*size_t(N_IDENTITIES)+size_t(c)];
                std::vector<float> pidv;
                softmax1d(row_id, pidv);
                p_id_claimed[size_t(i)] = pidv[size_t(claim[size_t(i)])];

                // std::vector<float> row_w(size_t(N_WINDOWS_PER_ID));
                std::vector<float> row_w(static_cast<size_t>(N_WINDOWS_PER_ID), 0.0f);

                for(int c=0;c<N_WINDOWS_PER_ID;c++) row_w[size_t(c)] = logits_w[size_t(i)*size_t(N_WINDOWS_PER_ID)+size_t(c)];
                std::vector<float> pwv;
                softmax1d(row_w, pwv);
                p_w_expected[size_t(i)] = pwv[size_t(expw[size_t(i)])];
            }

            std::vector<float> vb_in(size_t(B)*size_t(HIDDEN_DIM+4), 0.0f);
            std::vector<float> logits_v(size_t(B), 0.0f);

            for(int i=0;i<B;i++){
                float cid = float(claim[size_t(i)]) / float(std::max(1, N_IDENTITIES-1));
                float ew  = float(expw[size_t(i)])  / float(std::max(1, N_WINDOWS_PER_ID-1));
                for(int h=0;h<HIDDEN_DIM;h++){
                    vb_in[size_t(i)*size_t(HIDDEN_DIM+4) + size_t(h)] = cb[size_t(i)*size_t(HIDDEN_DIM) + size_t(h)];
                }
                vb_in[size_t(i)*size_t(HIDDEN_DIM+4) + size_t(HIDDEN_DIM + 0)] = cid;
                vb_in[size_t(i)*size_t(HIDDEN_DIM+4) + size_t(HIDDEN_DIM + 1)] = ew;
                vb_in[size_t(i)*size_t(HIDDEN_DIM+4) + size_t(HIDDEN_DIM + 2)] = p_id_claimed[size_t(i)];
                vb_in[size_t(i)*size_t(HIDDEN_DIM+4) + size_t(HIDDEN_DIM + 3)] = p_w_expected[size_t(i)];

                float s0 = p.b_beh(0);
                for(int j=0;j<HIDDEN_DIM+4;j++){
                    s0 += p.W_beh(0,j) * vb_in[size_t(i)*size_t(HIDDEN_DIM+4) + size_t(j)];
                }
                logits_v[size_t(i)] = s0;
            }

            std::vector<float> dlog_id, dlog_w, dlog_v;
            float loss_id = ce_loss_and_grad(logits_id, N_IDENTITIES, tid, pos, B, dlog_id);
            float loss_w  = ce_loss_and_grad(logits_w, N_WINDOWS_PER_ID, tw, pos, B, dlog_w);
            float loss_v  = bce_loss_and_grad(logits_v, yb, B, dlog_v);

            float loss = LAMBDA_ID*loss_id + LAMBDA_W*loss_w + LAMBDA_BCE*loss_v;
            epoch_loss_sum += double(loss) * double(B);

            Params grads = grads_zeros_like(p);

            for(int c=0;c<N_IDENTITIES;c++){
                float sb=0.0f;
                for(int i=0;i<B;i++) sb += dlog_id[size_t(i)*size_t(N_IDENTITIES) + size_t(c)];
                grads.b_id(c) += sb;

                for(int h=0;h<HIDDEN_DIM;h++){
                    float sW=0.0f;
                    for(int i=0;i<B;i++){
                        sW += dlog_id[size_t(i)*size_t(N_IDENTITIES) + size_t(c)] * cb[size_t(i)*size_t(HIDDEN_DIM) + size_t(h)];
                    }
                    grads.W_id(c,h) += sW;
                }
            }

            for(int c=0;c<N_WINDOWS_PER_ID;c++){
                float sb=0.0f;
                for(int i=0;i<B;i++) sb += dlog_w[size_t(i)*size_t(N_WINDOWS_PER_ID) + size_t(c)];
                grads.b_w(c) += sb;

                for(int j=0;j<3*HIDDEN_DIM;j++){
                    float sW=0.0f;
                    for(int i=0;i<B;i++){
                        sW += dlog_w[size_t(i)*size_t(N_WINDOWS_PER_ID) + size_t(c)] * feat_w[size_t(i)*size_t(3*HIDDEN_DIM) + size_t(j)];
                    }
                    grads.W_w(c,j) += sW;
                }
            }

            float sb=0.0f;
            for(int i=0;i<B;i++) sb += dlog_v[size_t(i)];
            grads.b_beh(0) += sb;

            for(int j=0;j<HIDDEN_DIM+4;j++){
                float sW=0.0f;
                for(int i=0;i<B;i++){
                    sW += dlog_v[size_t(i)] * vb_in[size_t(i)*size_t(HIDDEN_DIM+4) + size_t(j)];
                }
                grads.W_beh(0,j) += sW;
            }

            clip_grads(grads, CLIP_NORM);
            opt.step(p, grads, WEIGHT_DECAY, fr);
        }

        if(ep==2 || ep % std::max(1, EPOCHS_PHASE2/10) == 0){
            cout << "[Phase2] Epoch " << ep << "/" << EPOCHS_PHASE2
                 << " avg_loss=" << (epoch_loss_sum / double(ds.N))
                 << " lr=" << lr
                 << endl;
        }
    }
}

// ============================================================
// Save / Load model (binary)
// ============================================================
static void write_u32(std::ofstream& f, uint32_t x){ f.write(reinterpret_cast<const char*>(&x), sizeof(x)); }
static void write_u64(std::ofstream& f, uint64_t x){ f.write(reinterpret_cast<const char*>(&x), sizeof(x)); }
static uint32_t read_u32(std::ifstream& f){ uint32_t x; f.read(reinterpret_cast<char*>(&x), sizeof(x)); return x; }
static uint64_t read_u64(std::ifstream& f){ uint64_t x; f.read(reinterpret_cast<char*>(&x), sizeof(x)); return x; }

static void write_mat(std::ofstream& f, const Mat& m){
    write_u32(f, (uint32_t)m.R);
    write_u32(f, (uint32_t)m.C);
    write_u64(f, (uint64_t)m.a.size());
    f.write(reinterpret_cast<const char*>(m.a.data()), std::streamsize(m.a.size()*sizeof(float)));
}
static void write_vec(std::ofstream& f, const Vec& v){
    write_u32(f, (uint32_t)v.N);
    write_u64(f, (uint64_t)v.a.size());
    f.write(reinterpret_cast<const char*>(v.a.data()), std::streamsize(v.a.size()*sizeof(float)));
}

static void read_mat(std::ifstream& f, Mat& m){
    uint32_t R = read_u32(f);
    uint32_t C = read_u32(f);
    uint64_t sz = read_u64(f);
    m.R = int(R); m.C = int(C);
    m.a.assign(size_t(sz), 0.0f);
    f.read(reinterpret_cast<char*>(m.a.data()), std::streamsize(m.a.size()*sizeof(float)));
}
static void read_vec(std::ifstream& f, Vec& v){
    uint32_t N = read_u32(f);
    uint64_t sz = read_u64(f);
    v.N = int(N);
    v.a.assign(size_t(sz), 0.0f);
    f.read(reinterpret_cast<char*>(v.a.data()), std::streamsize(v.a.size()*sizeof(float)));
}

static bool save_model(const std::string& path, const Params& p){
    std::ofstream f(path, std::ios::binary);
    if(!f) return false;

    write_u32(f, 0x57414C4Bu); // magic
    write_u32(f, 1u);          // version

    write_u32(f, (uint32_t)VOCAB_SIZE);
    write_u32(f, (uint32_t)MS_DIM);
    write_u32(f, (uint32_t)SEQ_LEN);
    write_u32(f, (uint32_t)N_IDENTITIES);
    write_u32(f, (uint32_t)N_WINDOWS_PER_ID);
    write_u32(f, (uint32_t)HIDDEN_DIM);
    write_u32(f, (uint32_t)ATTN_DIM);
    write_u32(f, (uint32_t)MS_HID);
    write_u32(f, (uint32_t)INPUT_DIM);

    write_mat(f,p.W_z); write_mat(f,p.U_z); write_vec(f,p.b_z);
    write_mat(f,p.W_r); write_mat(f,p.U_r); write_vec(f,p.b_r);
    write_mat(f,p.W_h); write_mat(f,p.U_h); write_vec(f,p.b_h);

    write_mat(f,p.W_att); write_vec(f,p.v_att);

    write_mat(f,p.W_ms1); write_vec(f,p.b_ms1);
    write_mat(f,p.W_ms2); write_vec(f,p.b_ms2);

    write_mat(f,p.W_tok); write_vec(f,p.b_tok);

    write_mat(f,p.W_id); write_vec(f,p.b_id);
    write_mat(f,p.W_w);  write_vec(f,p.b_w);

    write_mat(f,p.W_beh); write_vec(f,p.b_beh);

    return true;
}

static bool load_model(const std::string& path, Params& p){
    std::ifstream f(path, std::ios::binary);
    if(!f) return false;

    uint32_t magic = read_u32(f);
    uint32_t ver   = read_u32(f);
    if(magic != 0x57414C4Bu || ver != 1u) return false;

    // consume constants (sanity)
    (void)read_u32(f); (void)read_u32(f); (void)read_u32(f);
    (void)read_u32(f); (void)read_u32(f); (void)read_u32(f);
    (void)read_u32(f); (void)read_u32(f); (void)read_u32(f);

    read_mat(f,p.W_z); read_mat(f,p.U_z); read_vec(f,p.b_z);
    read_mat(f,p.W_r); read_mat(f,p.U_r); read_vec(f,p.b_r);
    read_mat(f,p.W_h); read_mat(f,p.U_h); read_vec(f,p.b_h);

    read_mat(f,p.W_att); read_vec(f,p.v_att);

    read_mat(f,p.W_ms1); read_vec(f,p.b_ms1);
    read_mat(f,p.W_ms2); read_vec(f,p.b_ms2);

    read_mat(f,p.W_tok); read_vec(f,p.b_tok);

    read_mat(f,p.W_id); read_vec(f,p.b_id);
    read_mat(f,p.W_w);  read_vec(f,p.b_w);

    read_mat(f,p.W_beh); read_vec(f,p.b_beh);

    return true;
}

// ============================================================
// Verify (single sample) + gates: hard range break + PN pilot corr gate
// ============================================================
struct VerifyOut {
    bool ok=false;
    float p_valid=0;
    int id_pred=-1;
    int w_pred=-1;
    float pid=0;
    float pw=0;
    float l2ms=-1;
    float pilot_corr=0;
};

static VerifyOut verify_chain(
    const Params& p,
    const std::vector<int>& tokens,
    const std::vector<float>& meas,
    int claimed_id,
    int expected_w,
    const std::vector<float>* true_ms
){
    VerifyOut out;

    // HARD chain-break for out-of-range claims
    if (claimed_id < 0 || claimed_id >= N_IDENTITIES) return out;
    if (expected_w < 0 || expected_w >= N_WINDOWS_PER_ID) return out;

    // PN pilot correlation gate (fast early reject)
    int g_expected = claimed_id * N_WINDOWS_PER_ID + expected_w;
    out.pilot_corr = pilot_corr(meas, g_expected);
    bool pn_ok = (out.pilot_corr >= PILOT_CORR_MIN);
    // don’t early return — just remember pn_ok and include it as a gate later

    // out.pilot_corr = pilot_corr(meas, g_expected);
    // if (out.pilot_corr < PILOT_CORR_MIN) {
    //     out.p_valid = 0.0f;
    //     out.ok = false;
    //     return out;
    // }

    Mat X;
    build_X_backbone(tokens, meas, X);

    std::vector<float> Xb(size_t(SEQ_LEN)*size_t(INPUT_DIM), 0.0f);
    std::vector<float> Mb(size_t(SEQ_LEN), 1.0f);
    for(int t=0;t<SEQ_LEN;t++){
        for(int d=0; d<INPUT_DIM; d++){
            Xb[idx3(0,t,d,SEQ_LEN,INPUT_DIM)] = X(t,d);
        }
    }

    std::vector<float> H;
    GRUCache gcache;
    gru_forward_batch(p, Xb, Mb, 1, SEQ_LEN, INPUT_DIM, H, gcache);

    std::vector<float> ctx;
    AttnCache acache;
    attention_forward_batch(p, H, Mb, 1, SEQ_LEN, ctx, acache);

    std::vector<float> ms_hat, ms_hid;
    ms_head_forward(p, ctx, 1, ms_hat, ms_hid);

    std::vector<float> logits_id(size_t(N_IDENTITIES), 0.0f);
    for(int c=0;c<N_IDENTITIES;c++){
        float s0 = p.b_id(c);
        for(int h=0;h<HIDDEN_DIM;h++){
            s0 += p.W_id(c,h) * ctx[size_t(h)];
        }
        logits_id[size_t(c)] = s0;
    }
    std::vector<float> prob_id;
    softmax1d(logits_id, prob_id);
    out.id_pred = int(std::max_element(prob_id.begin(), prob_id.end()) - prob_id.begin());
    out.pid = prob_id[size_t(claimed_id)];

    // build hmean/hlast
    std::vector<float> hmean(size_t(HIDDEN_DIM),0.0f), hlast(size_t(HIDDEN_DIM),0.0f);
    float denom = 0.0f;
    for(int t=0;t<SEQ_LEN;t++) denom += Mb[idx2(0,t,SEQ_LEN)];
    denom += 1e-6f;
    for(int h=0;h<HIDDEN_DIM;h++){
        float sum=0.0f;
        for(int t=0;t<SEQ_LEN;t++){
            sum += Mb[idx2(0,t,SEQ_LEN)] * H[idx3(0,t,h,SEQ_LEN,HIDDEN_DIM)];
        }
        hmean[size_t(h)] = sum/denom;
    }
    int last = SEQ_LEN-1;
    for(int h=0;h<HIDDEN_DIM;h++) hlast[size_t(h)] = H[idx3(0,last,h,SEQ_LEN,HIDDEN_DIM)];

    std::vector<float> feat_w(size_t(3*HIDDEN_DIM), 0.0f);
    for(int h=0;h<HIDDEN_DIM;h++){
        feat_w[size_t(h)] = ctx[size_t(h)];
        feat_w[size_t(HIDDEN_DIM+h)] = hlast[size_t(h)];
        feat_w[size_t(2*HIDDEN_DIM+h)] = hmean[size_t(h)];
    }

    std::vector<float> logits_w(size_t(N_WINDOWS_PER_ID), 0.0f);
    for(int c=0;c<N_WINDOWS_PER_ID;c++){
        float s0 = p.b_w(c);
        for(int j=0;j<3*HIDDEN_DIM;j++){
            s0 += p.W_w(c,j) * feat_w[size_t(j)];
        }
        logits_w[size_t(c)] = s0;
    }
    std::vector<float> prob_w;
    softmax1d(logits_w, prob_w);
    out.w_pred = int(std::max_element(prob_w.begin(), prob_w.end()) - prob_w.begin());
    out.pw = prob_w[size_t(expected_w)];

    std::vector<float> vb_in(size_t(HIDDEN_DIM+4), 0.0f);
    for(int h=0;h<HIDDEN_DIM;h++) vb_in[size_t(h)] = ctx[size_t(h)];
    float cid = float(claimed_id)/float(std::max(1,N_IDENTITIES-1));
    float ew  = float(expected_w)/float(std::max(1,N_WINDOWS_PER_ID-1));
    vb_in[size_t(HIDDEN_DIM+0)] = cid;
    vb_in[size_t(HIDDEN_DIM+1)] = ew;
    vb_in[size_t(HIDDEN_DIM+2)] = out.pid;
    vb_in[size_t(HIDDEN_DIM+3)] = out.pw;

    float logit_v = p.b_beh(0);
    for(int j=0;j<HIDDEN_DIM+4;j++) logit_v += p.W_beh(0,j)*vb_in[size_t(j)];
    out.p_valid = sigmoid(logit_v);

    // bool c1 = (out.p_valid >= THRESH_P_VALID);
    // bool c2 = (out.id_pred == claimed_id);
    // bool c3 = (out.w_pred == expected_w);
    // bool c4 = (out.pid >= PID_MIN);
    // bool c5 = (out.pw >= PW_MIN);
    // out.ok = (c1 && c2 && c3 && c4 && c5);

    if(true_ms){
        std::vector<float> ms_hat_v(size_t(MS_DIM), 0.0f);
        for(int k=0;k<MS_DIM;k++) ms_hat_v[size_t(k)] = ms_hat[size_t(k)];
        out.l2ms = l2_vec(ms_hat_v, *true_ms);
    }
    // out.ok = (pn_ok && c1 && c2 && c3 && c4 && c5);
    bool c1 = (out.p_valid >= THRESH_P_VALID);
    bool c2 = (out.id_pred == claimed_id);
    bool c3 = (out.w_pred == expected_w);
    bool c4 = (out.pid >= PID_MIN);
    bool c5 = (out.pw >= PW_MIN);

    // optional: if you also want L2 gating
    // bool c6 = (!true_ms) ? true : (out.l2ms <= L2_MAX);

    out.ok = (pn_ok && c1 && c2 && c3 && c4 && c5);

    return out;
}

static void print_case(const std::string& title, const VerifyOut& r, int claimed_id, int expected_w) {
    cout << "\n=== " << title << " ===\n";
    cout << "CLAIMED: PeerID=" << claimed_id << " ExpectedWindow=" << expected_w << "\n";
    cout << "PRED:    id_pred=" << r.id_pred << " w_pred=" << r.w_pred << "\n";
    cout << "SCORES:  p_valid=" << r.p_valid << " pid=" << r.pid << " pw=" << r.pw << " pilot_corr=" << r.pilot_corr << "\n";
    if(r.l2ms >= 0) cout << "RECON:   L2(MS_hat,MS_true)=" << r.l2ms << "\n";
    cout << "GATES:   "
         << (r.pilot_corr >= PILOT_CORR_MIN ? "PN " : "PNx ")
         << (r.p_valid >= THRESH_P_VALID ? "pV " : "pVx ")
         << (r.id_pred == claimed_id ? "id " : "idx ")
         << (r.w_pred == expected_w ? "w " : "wx ")
         << (r.pid >= PID_MIN ? "pid " : "pidx ")
         << (r.pw >= PW_MIN ? "pw " : "pwx ")
         << "\n";
    cout << "FINAL:   OK=" << (r.ok ? "true":"false") << "\n";
}

// ============================================================
// Main
// ============================================================
int main() {
    cout << "WARL0K PIM C++ Demo (GRU+Attention, 2-phase, save/load, timing/sizing)\n";
    cout << "PN PILOT ENABLED amp=" << PILOT_AMP << " chips=" << SEQ_LEN
         << " corr_min=" << PILOT_CORR_MIN << "\n\n";

    // init global MS_all and A_base
    {
        XorShift32 rng(0xDEADBEEFu);
        MS_all = Mat(N_IDENTITIES, MS_DIM, 0.0f);
        for(int i=0;i<N_IDENTITIES;i++){
            for(int k=0;k<MS_DIM;k++){
                float u = rng.next_f01();
                MS_all(i,k) = 2.0f*u - 1.0f;
            }
        }
        A_base = Mat(SEQ_LEN, MS_DIM, 0.0f);
        for(int t=0;t<SEQ_LEN;t++){
            for(int k=0;k<MS_DIM;k++){
                A_base(t,k) = 0.8f * rng.next_norm();
            }
        }
    }

    Dataset ds = build_dataset();
    int positives = 0;
    for(float y: ds.Y_CLS) if(y > 0.5f) positives++;
    cout << "All samples: " << ds.N
         << " X shape: (" << ds.N << "," << SEQ_LEN << "," << INPUT_DIM << ")"
         << " Positives: " << positives << "\n\n";

    Params p = init_model(INPUT_DIM, 0xC0FFEEu);
    cout << "Model params: " << p.param_count()
         << " floats, size ~" << (double(p.bytes())/1024.0) << " KB\n";

    double t0 = now_seconds();
    train_phase1(p, ds);
    train_phase2(p, ds);
    double t1 = now_seconds();
    cout << "\nTRAINING TIME (total): " << (t1 - t0) << " sec\n";

    const std::string model_path = "warlok_pim_model.bin";
    if(!save_model(model_path, p)){
        cout << "ERROR: failed to save model\n";
        return 1;
    }
    cout << "Saved model -> " << model_path << " (" << file_size_bytes(model_path) << " bytes)\n";

    Params p2;
    if(!load_model(model_path, p2)){
        cout << "ERROR: failed to load model\n";
        return 1;
    }
    cout << "Loaded model OK. Params floats=" << p2.param_count()
         << " size ~" << (double(p2.bytes())/1024.0) << " KB\n";

    int id_eval = 0;
    int w_eval  = 3;
    std::vector<float> ms_eval(size_t(MS_DIM), 0.0f);
    for(int k=0;k<MS_DIM;k++) ms_eval[size_t(k)] = MS_all(id_eval,k);

    int g_true = id_eval * N_WINDOWS_PER_ID + w_eval;
    std::vector<int> toks;
    std::vector<float> meas;
    generate_os_chain(ms_eval, g_true, toks, meas);

    const int INFER_ITERS = 100;
    double ti0 = now_seconds();
    VerifyOut last;
    volatile float sink = 0.0f;
    for(int i=0;i<INFER_ITERS;i++){
        last = verify_chain(p2, toks, meas, id_eval, w_eval, &ms_eval);
        sink += last.p_valid; // prevents optimization
    }
    cout << "sink=" << sink << "\n";

    // for(int i=0;i<INFER_ITERS;i++){
    //     last = verify_chain(p2, toks, meas, id_eval, w_eval, &ms_eval);
    // }
    double ti1 = now_seconds();
    double per = (ti1 - ti0) / double(INFER_ITERS);
    cout << "\nINFERENCE TIME: total=" << (ti1-ti0) << " sec"
         << " per_call=" << per*1e6 << " us (" << per*1e3 << " ms)\n";

    {
        auto r = verify_chain(p2, toks, meas, id_eval, w_eval, &ms_eval);
        print_case("LEGIT (intact chain)", r, id_eval, w_eval);

        // shuffled
        std::vector<int> idxs(SEQ_LEN);
        std::iota(idxs.begin(), idxs.end(), 0);
        XorShift32 rsh(0x7777u);
        shuffle_inplace(idxs, rsh);
        std::vector<int> toks_s(SEQ_LEN);
        std::vector<float> meas_s(SEQ_LEN);
        for(int t=0;t<SEQ_LEN;t++){ toks_s[size_t(t)]=toks[size_t(idxs[size_t(t)])]; meas_s[size_t(t)]=meas[size_t(idxs[size_t(t)])]; }
        r = verify_chain(p2, toks_s, meas_s, id_eval, w_eval, &ms_eval);
        print_case("TAMPER: SHUFFLED (replay-ish)", r, id_eval, w_eval);

        // truncated
        int Ltr = SEQ_LEN/2;
        std::vector<int> toks_t(SEQ_LEN, 0);
        std::vector<float> meas_t(SEQ_LEN, 0.0f);
        for(int t=0;t<Ltr;t++){ toks_t[size_t(t)]=toks[size_t(t)]; meas_t[size_t(t)]=meas[size_t(t)]; }
        r = verify_chain(p2, toks_t, meas_t, id_eval, w_eval, &ms_eval);
        print_case("TAMPER: TRUNCATED (dropped steps)", r, id_eval, w_eval);

        // wrong window
        int wrong_w = (w_eval + 7) % N_WINDOWS_PER_ID;
        int g_wrong = id_eval * N_WINDOWS_PER_ID + wrong_w;
        std::vector<int> toks_w;
        std::vector<float> meas_w;
        generate_os_chain(ms_eval, g_wrong, toks_w, meas_w);
        r = verify_chain(p2, toks_w, meas_w, id_eval, w_eval, &ms_eval);
        print_case("TAMPER: WRONG WINDOW (counter drift)", r, id_eval, w_eval);

        // wrong identity
        int other_id = (id_eval + 1) % N_IDENTITIES;
        int other_w  = 13;
        int g_other  = other_id * N_WINDOWS_PER_ID + other_w;
        std::vector<float> ms_other(size_t(MS_DIM), 0.0f);
        for(int k=0;k<MS_DIM;k++) ms_other[size_t(k)] = MS_all(other_id,k);
        std::vector<int> toks_o;
        std::vector<float> meas_o;
        generate_os_chain(ms_other, g_other, toks_o, meas_o);
        r = verify_chain(p2, toks_o, meas_o, id_eval, w_eval, &ms_eval);
        print_case("TAMPER: WRONG IDENTITY (impersonation)", r, id_eval, w_eval);

        // hard break demo: out-of-range window
        r = verify_chain(p2, toks, meas, id_eval, 999, &ms_eval);
        print_case("HARD BREAK: expected_w out of range", r, id_eval, 999);
    }

    cout << "\nDone.\n";
    return 0;
}

