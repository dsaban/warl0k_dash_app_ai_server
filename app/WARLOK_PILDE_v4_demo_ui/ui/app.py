
import streamlit as st
import requests
import networkx as nx
import matplotlib.pyplot as plt

API = "http://127.0.0.1:8000"

st.title("WARLOK PILDE DAG Demo")

payload = st.text_input("Payload", "test")
parents = st.text_input("Parents (comma-separated)", "")

if st.button("Create Block"):
    p = [x.strip() for x in parents.split(",") if x]
    res = requests.post(f"{API}/create_block", json={
        "parents": p,
        "payload": payload,
        "metadata": {}
    })
    st.write(res.json())

if st.button("Show DAG"):
    res = requests.get(f"{API}/dag")
    data = res.json()

    G = nx.DiGraph()

    for h, node in data.items():
        G.add_node(h[:6])
        for p in node["parents"]:
            G.add_edge(p[:6], h[:6])

    fig, ax = plt.subplots()
    nx.draw(G, with_labels=True, node_size=1500, font_size=8)
    st.pyplot(fig)
