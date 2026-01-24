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

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include <cmath>

using std::cerr;
using std::cout;
using std::endl;

static inline float sigmoidf(float x) {
  return 1.0f / (1.0f + std::exp(-x));
}

static inline float tanhf_fast(float x) { return std::tanh(x); }

static inline float clampf(float x, float lo, float hi) {
  return std::max(lo, std::min(hi, x));
}

static void softmax1d(std::vector<float>& v) {
  float mx = -1e30f;
  for (float x : v) mx = std::max(mx, x);
  float sum = 0.0f;
  for (float& x : v) {
    x = std::exp(x - mx);
    sum += x;
  }
  float inv = 1.0f / (sum + 1e-12f);
  for (float& x : v) x *= inv;
}

static float l2_norm(const std::vector<float>& a, const std::vector<float>& b) {
  float s = 0.0f;
  for (size_t i = 0; i < a.size(); i++) {
    float d = a[i] - b[i];
    s += d * d;
  }
  return std::sqrt(s);
}

// Simple xorshift32 for deterministic PN pilot
static inline uint32_t xorshift32(uint32_t& x) {
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  return x;
}

struct Tensor {
  uint32_t rows = 0, cols = 0;
  std::vector<float> data; // rows*cols
};

static size_t tensor_bytes(const Tensor& t) {
  return sizeof(float) * t.data.size();
}

struct Model {
  int VOCAB_SIZE = 16;
  int MS_DIM = 8;
  int SEQ_LEN = 20;

  int N_IDENTITIES = 6;
  int N_WINDOWS_PER_ID = 48;

  int HIDDEN_DIM = 48;
  int ATTN_DIM = 24;
  int MS_HID = 32;

  Tensor W_z, U_z, b_z;
  Tensor W_r, U_r, b_r;
  Tensor W_h, U_h, b_h;

  Tensor W_att, v_att;

  Tensor W_ms1, b_ms1, W_ms2, b_ms2;
  Tensor W_tok, b_tok;

  Tensor W_id, b_id;
  Tensor W_w, b_w;
  Tensor W_beh, b_beh;

  size_t bytes_total() const {
    size_t b = 0;
    const Tensor* all[] = {
        &W_z,&U_z,&b_z,&W_r,&U_r,&b_r,&W_h,&U_h,&b_h,
        &W_att,&v_att,&W_ms1,&b_ms1,&W_ms2,&b_ms2,&W_tok,&b_tok,
        &W_id,&b_id,&W_w,&b_w,&W_beh,&b_beh
    };
    for (auto* t : all) b += tensor_bytes(*t);
    return b;
  }
};

static bool read_u32(std::ifstream& f, uint32_t& out) {
  f.read(reinterpret_cast<char*>(&out), sizeof(out));
  return bool(f);
}
static bool read_i32(std::ifstream& f, int32_t& out) {
  f.read(reinterpret_cast<char*>(&out), sizeof(out));
  return bool(f);
}

static bool read_tensor(std::ifstream& f, Tensor& t) {
  uint32_t ndim = 0, r = 0, c = 0;
  if (!read_u32(f, ndim)) return false;
  if (!read_u32(f, r)) return false;
  if (!read_u32(f, c)) return false;
  if (ndim != 1 && ndim != 2) return false;
  t.rows = r;
  t.cols = c;
  size_t n = size_t(r) * size_t(c);
  t.data.resize(n);
  f.read(reinterpret_cast<char*>(t.data.data()), n * sizeof(float));
  return bool(f);
}

static bool load_model(const std::string& path, Model& m) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return false;

  uint32_t magic = 0, version = 0;
  if (!read_u32(f, magic)) return false;
  if (!read_u32(f, version)) return false;
  if (magic != 0x4B4C5257u || version != 1u) { // "WRLK"
    cerr << "Bad model header (magic/version mismatch).\n";
    return false;
  }

  int32_t tmp = 0;
  read_i32(f, tmp); m.VOCAB_SIZE = tmp;
  read_i32(f, tmp); m.MS_DIM = tmp;
  read_i32(f, tmp); m.SEQ_LEN = tmp;
  read_i32(f, tmp); m.N_IDENTITIES = tmp;
  read_i32(f, tmp); m.N_WINDOWS_PER_ID = tmp;
  read_i32(f, tmp); m.HIDDEN_DIM = tmp;
  read_i32(f, tmp); m.ATTN_DIM = tmp;
  read_i32(f, tmp); m.MS_HID = tmp;

  Tensor* all[] = {
      &m.W_z,&m.U_z,&m.b_z,
      &m.W_r,&m.U_r,&m.b_r,
      &m.W_h,&m.U_h,&m.b_h,
      &m.W_att,&m.v_att,
      &m.W_ms1,&m.b_ms1,&m.W_ms2,&m.b_ms2,
      &m.W_tok,&m.b_tok,
      &m.W_id,&m.b_id,
      &m.W_w,&m.b_w,
      &m.W_beh,&m.b_beh
  };
  for (auto* t : all) {
    if (!read_tensor(f, *t)) return false;
  }
  return true;
}

// Matvec: y = A(rows x cols) * x(cols)
static void matvec(const Tensor& A, const std::vector<float>& x, std::vector<float>& y) {
  y.assign(A.rows, 0.0f);
  for (uint32_t i = 0; i < A.rows; i++) {
    float s = 0.0f;
    const float* row = &A.data[size_t(i) * A.cols];
    for (uint32_t j = 0; j < A.cols; j++) s += row[j] * x[j];
    y[i] = s;
  }
}

static void add_inplace(std::vector<float>& a, const std::vector<float>& b) {
  for (size_t i = 0; i < a.size(); i++) a[i] += b[i];
}

static void add_bias_inplace(std::vector<float>& a, const Tensor& b) {
  for (size_t i = 0; i < a.size(); i++) a[i] += b.data[i];
}

static void sigmoid_vec(std::vector<float>& v) {
  for (float& x : v) x = sigmoidf(x);
}
static void tanh_vec(std::vector<float>& v) {
  for (float& x : v) x = tanhf_fast(x);
}

static std::vector<float> hadamard(const std::vector<float>& a, const std::vector<float>& b) {
  std::vector<float> y(a.size());
  for (size_t i = 0; i < a.size(); i++) y[i] = a[i] * b[i];
  return y;
}

static void build_backbone(const Model& m,
                           const std::vector<int>& tokens,
                           const std::vector<float>& meas,
                           std::vector<float>& X_flat,
                           std::vector<float>& M)
{
  const int T = m.SEQ_LEN;
  const int D = m.VOCAB_SIZE + 2;
  X_flat.assign(size_t(T) * size_t(D), 0.0f);
  M.assign(T, 1.0f);

  for (int t = 0; t < T; t++) {
    const int tok = tokens[t];
    X_flat[size_t(t)*D + tok] = 1.0f;
    X_flat[size_t(t)*D + m.VOCAB_SIZE] = meas[t];
    X_flat[size_t(t)*D + m.VOCAB_SIZE + 1] = float(t) / float(std::max(1, T-1));
  }
}

static void pn_pilot(int window_global_id, int seq_len, float amp, std::vector<float>& pilot) {
  pilot.resize(seq_len);
  uint32_t st = (uint32_t)(window_global_id * 9176 + 11);
  float mean = 0.0f;
  for (int i = 0; i < seq_len; i++) {
    uint32_t r = xorshift32(st);
    float chip = (r & 1u) ? 1.0f : -1.0f;
    pilot[i] = amp * chip;
    mean += pilot[i];
  }
  mean /= float(seq_len);
  for (int i = 0; i < seq_len; i++) pilot[i] -= mean;
}

static void infer_embeddings(const Model& m,
                             const std::vector<float>& X_flat,
                             const std::vector<float>& M,
                             std::vector<float>& ctx,
                             std::vector<float>& h_last,
                             std::vector<float>& h_mean)
{
  const int T = m.SEQ_LEN;
  const int D = m.VOCAB_SIZE + 2;
  const int H = m.HIDDEN_DIM;

  std::vector<float> hs(size_t(T) * size_t(H), 0.0f);
  std::vector<float> h_prev(H, 0.0f);

  std::vector<float> x(D), a_z(H), a_r(H), z(H), r(H), a_h(H), htil(H), h(H);
  std::vector<float> tmp(H);

  for (int t = 0; t < T; t++) {
    for (int j = 0; j < D; j++) x[j] = X_flat[size_t(t)*D + j];
    float mt = M[t];

    matvec(m.W_z, x, a_z);
    matvec(m.U_z, h_prev, tmp);
    add_inplace(a_z, tmp);
    add_bias_inplace(a_z, m.b_z);
    z = a_z; sigmoid_vec(z);

    matvec(m.W_r, x, a_r);
    matvec(m.U_r, h_prev, tmp);
    add_inplace(a_r, tmp);
    add_bias_inplace(a_r, m.b_r);
    r = a_r; sigmoid_vec(r);

    auto rhprev = hadamard(r, h_prev);
    matvec(m.W_h, x, a_h);
    matvec(m.U_h, rhprev, tmp);
    add_inplace(a_h, tmp);
    add_bias_inplace(a_h, m.b_h);
    htil = a_h; tanh_vec(htil);

    for (int i = 0; i < H; i++) {
      float hv = (1.0f - z[i]) * h_prev[i] + z[i] * htil[i];
      h[i] = mt * hv + (1.0f - mt) * h_prev[i];
    }

    std::memcpy(&hs[size_t(t)*H], h.data(), sizeof(float) * H);
    h_prev = h;
  }

  h_last = h_prev;

  h_mean.assign(H, 0.0f);
  float denom = 0.0f;
  for (int t = 0; t < T; t++) {
    float mt = M[t];
    denom += mt;
    const float* ht = &hs[size_t(t)*H];
    for (int i = 0; i < H; i++) h_mean[i] += mt * ht[i];
  }
  denom = std::max(denom, 1e-6f);
  for (int i = 0; i < H; i++) h_mean[i] /= denom;

  // Attention
  std::vector<float> scores(T, -1e9f);
  std::vector<float> u(m.ATTN_DIM, 0.0f);

  for (int t = 0; t < T; t++) {
    if (M[t] <= 0.0f) continue;
    std::vector<float> h_t(H);
    std::memcpy(h_t.data(), &hs[size_t(t)*H], sizeof(float)*H);
    matvec(m.W_att, h_t, u);
    tanh_vec(u);
    float s = 0.0f;
    for (int a = 0; a < m.ATTN_DIM; a++) s += m.v_att.data[a] * u[a];
    scores[t] = s;
  }

  float mx = -1e30f;
  for (int t = 0; t < T; t++) if (M[t] > 0.0f) mx = std::max(mx, scores[t]);
  float sum = 0.0f;
  std::vector<float> alpha(T, 0.0f);
  for (int t = 0; t < T; t++) {
    if (M[t] <= 0.0f) continue;
    float e = std::exp(scores[t] - mx);
    alpha[t] = e;
    sum += e;
  }
  float inv = 1.0f / (sum + 1e-12f);
  for (int t = 0; t < T; t++) alpha[t] *= inv;

  ctx.assign(H, 0.0f);
  for (int t = 0; t < T; t++) {
    const float* ht = &hs[size_t(t)*H];
    float at = alpha[t];
    for (int i = 0; i < H; i++) ctx[i] += at * ht[i];
  }
}

static void infer_ms(const Model& m, const std::vector<float>& ctx, std::vector<float>& ms_hat) {
  std::vector<float> h1(m.MS_HID, 0.0f);
  matvec(m.W_ms1, ctx, h1);
  for (int i = 0; i < m.MS_HID; i++) h1[i] += m.b_ms1.data[i];
  tanh_vec(h1);

  matvec(m.W_ms2, h1, ms_hat);
  for (int i = 0; i < m.MS_DIM; i++) ms_hat[i] += m.b_ms2.data[i];
}

struct VerifyOut {
  bool ok = false;
  float p_valid = 0.0f;
  int id_pred = -1;
  int w_pred = -1;
  float pid = 0.0f;
  float pw = 0.0f;
  std::vector<float> ms_hat;
};

static VerifyOut verify(const Model& m,
                        const std::vector<int>& tokens,
                        const std::vector<float>& meas,
                        int claimed_id,
                        int expected_w,
                        float thresh_p_valid=0.70f,
                        float pid_min=0.70f,
                        float pw_min=0.70f)
{
  VerifyOut out;
  std::vector<float> X, M;
  build_backbone(m, tokens, meas, X, M);

  std::vector<float> ctx, h_last, h_mean;
  infer_embeddings(m, X, M, ctx, h_last, h_mean);

  infer_ms(m, ctx, out.ms_hat);

  std::vector<float> logits_id(m.N_IDENTITIES, 0.0f);
  matvec(m.W_id, ctx, logits_id);
  for (int i = 0; i < m.N_IDENTITIES; i++) logits_id[i] += m.b_id.data[i];

  auto prob_id = logits_id;
  softmax1d(prob_id);
  out.id_pred = int(std::max_element(prob_id.begin(), prob_id.end()) - prob_id.begin());
  out.pid = prob_id[claimed_id];

  std::vector<float> feat_w;
  feat_w.reserve(size_t(3)*size_t(m.HIDDEN_DIM));
  feat_w.insert(feat_w.end(), ctx.begin(), ctx.end());
  feat_w.insert(feat_w.end(), h_last.begin(), h_last.end());
  feat_w.insert(feat_w.end(), h_mean.begin(), h_mean.end());

  std::vector<float> logits_w(m.N_WINDOWS_PER_ID, 0.0f);
  matvec(m.W_w, feat_w, logits_w);
  for (int i = 0; i < m.N_WINDOWS_PER_ID; i++) logits_w[i] += m.b_w.data[i];

  auto prob_w = logits_w;
  softmax1d(prob_w);
  out.w_pred = int(std::max_element(prob_w.begin(), prob_w.end()) - prob_w.begin());
  out.pw = prob_w[expected_w];

  float cid = float(claimed_id) / float(std::max(1, m.N_IDENTITIES - 1));
  float ew  = float(expected_w) / float(std::max(1, m.N_WINDOWS_PER_ID - 1));

  std::vector<float> vb_in;
  vb_in.reserve(size_t(m.HIDDEN_DIM) + 4);
  vb_in.insert(vb_in.end(), ctx.begin(), ctx.end());
  vb_in.push_back(cid);
  vb_in.push_back(ew);
  vb_in.push_back(out.pid);
  vb_in.push_back(out.pw);

  float logit_v = 0.0f;
  for (uint32_t j = 0; j < m.W_beh.cols; j++) logit_v += m.W_beh.data[j] * vb_in[j];
  logit_v += m.b_beh.data[0];
  out.p_valid = sigmoidf(logit_v);

  bool ok = true;
  ok = ok && (out.p_valid >= thresh_p_valid);
  ok = ok && (out.id_pred == claimed_id);
  ok = ok && (out.w_pred == expected_w);
  ok = ok && (out.pid >= pid_min);
  ok = ok && (out.pw >= pw_min);
  out.ok = ok;

  return out;
}

static void print_case(const std::string& title,
                       const VerifyOut& r,
                       int claimed_id,
                       int expected_w,
                       const std::vector<float>* ms_true=nullptr)
{
  cout << "\n=== " << title << " ===\n";
  cout << "CLAIMED ENTITY:   PeerID=" << claimed_id << "   ExpectedWindow=" << expected_w << "\n";
  cout << "MODEL PREDICTION: id_pred=" << r.id_pred << "  w_pred=" << r.w_pred << "\n";
  cout << std::fixed << std::setprecision(6);
  cout << "SCORES: p_valid=" << r.p_valid
       << "  pid(claim)=" << r.pid
       << "  pw(expected)=" << r.pw << "\n";
  if (ms_true) {
    cout << "RECON:  L2(MS_hat, MS_true)=" << l2_norm(r.ms_hat, *ms_true) << "\n";
  }
  cout << "FINAL DECISION: OK=" << (r.ok ? "True" : "False") << "\n";
}

int main(int argc, char** argv) {
  if (argc < 2) {
    cerr << "Usage: " << argv[0] << " model.bin\n";
    return 2;
  }
  const std::string model_path = argv[1];

  Model m;

  auto t0 = std::chrono::high_resolution_clock::now();
  bool ok = load_model(model_path, m);
  auto t1 = std::chrono::high_resolution_clock::now();

  if (!ok) {
    cerr << "Failed to load model: " << model_path << "\n";
    return 1;
  }

  double load_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  size_t bytes = m.bytes_total();

  cout << "Loaded model: " << model_path << "\n";
  cout << "Model parameter bytes: " << bytes << " bytes ("
       << (bytes / 1024.0) << " KiB)\n";
  cout << "Load time: " << std::fixed << std::setprecision(3) << load_ms << " ms\n";

  const int claimed_id = 0;
  const int expected_w = 5;

  const int T = m.SEQ_LEN;
  std::vector<int> tokens(T, 0);
  std::vector<float> meas(T, 0.0f);

  std::vector<float> pilot;
  int window_global = claimed_id * m.N_WINDOWS_PER_ID + expected_w;
  pn_pilot(window_global, T, 0.55f, pilot);

  for (int t = 0; t < T; t++) {
    float v = pilot[t];
    meas[t] = v;
    float scaled = clampf((v + 3.0f) / 6.0f, 0.0f, 0.999999f);
    tokens[t] = int(scaled * float(m.VOCAB_SIZE));
  }

  auto ti0 = std::chrono::high_resolution_clock::now();
  VerifyOut r_legit = verify(m, tokens, meas, claimed_id, expected_w);
  auto ti1 = std::chrono::high_resolution_clock::now();
  double infer_ms = std::chrono::duration<double, std::milli>(ti1 - ti0).count();

  print_case("LEGIT (toy sample, PN pilot only)", r_legit, claimed_id, expected_w, nullptr);
  cout << "Inference time (single sample): " << std::fixed << std::setprecision(6) << infer_ms << " ms\n";

  cout << "\nNOTE:\n"
       << "  This C++ demo uses a toy-generated sample.\n"
       << "  For real validation (matching Python), export tokens+meas from Python\n"
       << "  for each scenario (legit/shuffled/truncated/wrong-window/wrong-id)\n"
       << "  and feed them into this verifier.\n";

  return 0;
}

/*
======================== PYTHON EXPORTER (add to your training script) ========================

import struct
import numpy as np

def _write_tensor(f, arr):
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        ndim, r, c = 1, arr.shape[0], 1
        flat = arr.reshape(-1)
    elif arr.ndim == 2:
        ndim, r, c = 2, arr.shape[0], arr.shape[1]
        flat = arr.reshape(-1)
    else:
        raise ValueError(arr.ndim)
    f.write(struct.pack("<III", ndim, r, c))
    f.write(flat.tobytes(order="C"))

def export_model_bin(path, p, cfg):
    with open(path, "wb") as f:
        f.write(struct.pack("<II", 0x4B4C5257, 1))  # "WRLK", v1
        header = [
            cfg["VOCAB_SIZE"], cfg["MS_DIM"], cfg["SEQ_LEN"],
            cfg["N_IDENTITIES"], cfg["N_WINDOWS_PER_ID"],
            cfg["HIDDEN_DIM"], cfg["ATTN_DIM"], cfg["MS_HID"]
        ]
        f.write(struct.pack("<" + "i"*len(header), *header))

        order = [
            "W_z","U_z","b_z",
            "W_r","U_r","b_r",
            "W_h","U_h","b_h",
            "W_att","v_att",
            "W_ms1","b_ms1","W_ms2","b_ms2",
            "W_tok","b_tok",
            "W_id","b_id",
            "W_w","b_w",
            "W_beh","b_beh"
        ]
        for k in order:
            _write_tensor(f, p[k])

# Example:
cfg = dict(VOCAB_SIZE=VOCAB_SIZE, MS_DIM=MS_DIM, SEQ_LEN=SEQ_LEN,
           N_IDENTITIES=N_IDENTITIES, N_WINDOWS_PER_ID=N_WINDOWS_PER_ID,
           HIDDEN_DIM=HIDDEN_DIM, ATTN_DIM=ATTN_DIM, MS_HID=MS_HID)
export_model_bin("model.bin", p, cfg)

==============================================================================================
*/
