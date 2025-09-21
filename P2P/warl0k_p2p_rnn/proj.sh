# shellcheck disable=SC2287
warlok_model_demo/
├── README.md
├── requirements.txt
├── warlok/                      # core WARL0K code (lightweight)
│   ├── __init__.py
│   ├── crypto_utils.py
│   ├── hub.py
│   ├── peer.py
│   └── socket_net.py
├── model/                       # your uploaded RNN + wrapper
│   ├── rnn_model.py             # the uploaded model classes (from your file)
│   └── model_wrapper.py         # small wrapper to adapt obf strings -> train/predict
├── demo_model.py                # runnable demo that uses sockets and trains the model each session
└── LICENSE
