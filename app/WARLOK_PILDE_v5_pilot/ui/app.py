
import streamlit as st, requests, networkx as nx, matplotlib.pyplot as plt
API="http://127.0.0.1:8000"

def safe_post(ep,p):
    try:
        r=requests.post(API+ep,json=p,timeout=3)
        return r.json()
    except Exception as e:
        return {"status":"error","message":str(e)}

st.title("PILDE Demo")

try:
    if requests.get(API+"/health").status_code==200:
        st.success("API OK")
except:
    st.error("API DOWN")

payload=st.text_input("Payload","test")
parents=st.text_input("Parents","")

if st.button("Create"):
    res=safe_post("/create_block",{"parents":[x for x in parents.split(",") if x],"payload":payload})
    st.write(res)

if st.button("Show DAG"):
    try:
        data=requests.get(API+"/dag").json()["nodes"]
        G=nx.DiGraph()
        for h,n in data.items():
            G.add_node(h[:6])
            for p in n["parents"]:
                G.add_edge(p[:6],h[:6])
        fig,ax=plt.subplots()
        nx.draw(G,with_labels=True)
        st.pyplot(fig)
    except Exception as e:
        st.error(str(e))
