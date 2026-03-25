
import streamlit as st,requests,networkx as nx,matplotlib.pyplot as plt
API="http://127.0.0.1:8000"
st.set_page_config(layout="wide")
st.title("WARLOK PILDE Console")

# System
col1,col2,col3=st.columns(3)
try:
 health=requests.get(API+"/health").json()
 col1.metric("System",health["status"])
except:
 col1.metric("System","DOWN")

events=requests.get(API+"/events").json()["events"]
col2.metric("Events",len(events))
col3.metric("HSM Ops",len([e for e in events if "HSM" in e["type"]]))

st.divider()

# Create
payload=st.text_input("Payload","genesis")
parents=st.text_input("Parents","")
if st.button("Create"):
 p=[x.strip() for x in parents.split(",") if x]
 st.write(requests.post(API+"/create_block",json={"parents":p,"payload":payload}).json())

# Events
st.subheader("Timeline")
for e in reversed(events[-10:]):
 st.write(e)

# DAG
st.subheader("DAG")
nodes=requests.get(API+"/dag").json()["nodes"]
G=nx.DiGraph()
for h,n in nodes.items():
 G.add_node(h[:6])
 for p in n["parents"]:
  G.add_edge(p[:6],h[:6])
fig,ax=plt.subplots()
nx.draw(G,with_labels=True)
st.pyplot(fig)

# Inspector
sel=st.text_input("Inspect Hash")
if sel:
 if sel in nodes:
  st.json(nodes[sel])
  st.write(requests.post(API+"/validate",json={"hash":sel}).json())
 else:
  st.error("Not found")
