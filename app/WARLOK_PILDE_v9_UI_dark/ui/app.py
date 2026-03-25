
import streamlit as st,requests,networkx as nx,matplotlib.pyplot as plt
API="http://127.0.0.1:8000"
st.set_page_config(layout="wide")
st.title("🛡️ WARLOK DEMO MODE")
page=st.sidebar.radio("Navigation",["Dashboard","Execution","Graph","Inspector","Attack"])
def fetch():
    try:
        return (requests.get(API+"/dag").json()["nodes"],requests.get(API+"/events").json()["events"])
    except:
        return {},[]
dag,events=fetch()
if page=="Dashboard":
    st.button("Run Pipeline",on_click=lambda:requests.post(API+"/pipeline"))
elif page=="Execution":
    for e in reversed(events[-30:]):
        st.write(e)
elif page=="Graph":
    G=nx.DiGraph()
    colors=[]
    for h,n in dag.items():
        G.add_node(h[:6])
        colors.append("green" if n["status"]=="ACTIVE" else "red")
        for p in n["parents"]:
            G.add_edge(p[:6],h[:6])
    fig,ax=plt.subplots()
    nx.draw(G,node_color=colors,with_labels=True)
    st.pyplot(fig)
elif page=="Inspector":
    sel=st.text_input("Hash")
    if sel in dag:
        st.json(dag[sel])
        st.write(requests.post(API+"/validate",json={"hash":sel}).json())
elif page=="Attack":
    sel=st.text_input("Target Hash")
    if st.button("Tamper"):
        requests.post(API+"/attack/tamper",json={"hash":sel})
    if st.button("Break Signature"):
        requests.post(API+"/attack/signature",json={"hash":sel})
