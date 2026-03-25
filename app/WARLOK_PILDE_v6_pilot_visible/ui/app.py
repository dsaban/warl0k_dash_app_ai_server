
import streamlit as st,requests,networkx as nx,matplotlib.pyplot as plt
API="http://127.0.0.1:8000"

st.title("PILDE Observatory")

try:
    if requests.get(API+"/health").status_code==200:
        st.success("API OK")
except:
    st.error("API DOWN")

payload=st.text_input("Payload","genesis")
parents=st.text_input("Parents","")

if st.button("Create"):
    p=[x.strip() for x in parents.split(",") if x]
    st.write(requests.post(API+"/create_block",json={"parents":p,"payload":payload}).json())

st.subheader("Events")
try:
    ev=requests.get(API+"/events").json()["events"]
    for e in reversed(ev[-20:]):
        st.write(e)
except: st.write("no events")

st.subheader("DAG")
try:
    nodes=requests.get(API+"/dag").json()["nodes"]
    G=nx.DiGraph()
    for h,n in nodes.items():
        G.add_node(h[:6])
        for p in n["parents"]:
            G.add_edge(p[:6],h[:6])
    fig,ax=plt.subplots()
    nx.draw(G,with_labels=True)
    st.pyplot(fig)
except: st.write("no dag")

h=st.text_input("validate hash")
if st.button("Validate"):
    st.write(requests.post(API+"/validate",json={"hash":h}).json())
