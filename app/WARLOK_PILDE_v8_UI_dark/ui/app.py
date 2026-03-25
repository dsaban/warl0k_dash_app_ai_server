
import streamlit as st,requests,networkx as nx,matplotlib.pyplot as plt

API="http://127.0.0.1:8000"

st.set_page_config(layout="wide")
st.title("🛡️ WARLOK PILDE Console")

page=st.sidebar.radio("Navigation",[
 "Dashboard","Execution Flow","DAG View","Node Inspector","Security","Archive"
])

def fetch():
    try:
        return (
            requests.get(API+"/health").json(),
            requests.get(API+"/dag").json()["nodes"],
            requests.get(API+"/events").json()["events"]
        )
    except:
        return None,{},[]

health,dag,events=fetch()

if page=="Dashboard":
    col1,col2,col3=st.columns(3)
    col1.metric("System",health["status"] if health else "DOWN")
    col2.metric("Events",len(events))
    col3.metric("HSM Ops",len([e for e in events if "HSM" in e["type"]]))
    st.divider()
    payload=st.text_input("Payload","genesis")
    parents=st.text_input("Parents")
    if st.button("Create"):
        p=[x.strip() for x in parents.split(",") if x]
        st.success(requests.post(API+"/create_block",json={"parents":p,"payload":payload}).json())

elif page=="Execution Flow":
    for e in reversed(events[-20:]):
        st.write(e)

elif page=="DAG View":
    G=nx.DiGraph()
    for h,n in dag.items():
        G.add_node(h[:6])
        for p in n["parents"]:
            G.add_edge(p[:6],h[:6])
    fig,ax=plt.subplots()
    nx.draw(G,with_labels=True,node_size=2000)
    st.pyplot(fig)

elif page=="Node Inspector":
    sel=st.text_input("Node Hash")
    if sel and sel in dag:
        st.json(dag[sel])
        val=requests.post(API+"/validate",json={"hash":sel}).json()
        st.write(val)
    elif sel:
        st.error("Not found")

elif page=="Security":
    for e in events:
        if "HSM" in e["type"]:
            st.write(e)

elif page=="Archive":
    for h,n in dag.items():
        st.write(n)
