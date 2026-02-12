import networkx as nx
import matplotlib.pyplot as plt

def render_invoice_graph(st, invoice, checks):
    G = nx.DiGraph()
    inv_node = f"Invoice\n{invoice.get('invoice_id','')}\n${invoice.get('amount_usd','')}"
    ven_node = f"Vendor\n{invoice.get('vendor_id','')}"
    G.add_edge(ven_node, inv_node)

    for c in checks:
        node = f"{c['cid']}\n{c['title']}\n[{c['status']}]"
        G.add_edge(inv_node, node)
        act = f"Action\n{c['recommendation'][:60]}{'…' if len(c['recommendation'])>60 else ''}"
        G.add_edge(node, act)

    pos = nx.spring_layout(G, seed=5, k=0.8)

    def color_for(n):
        if n.startswith("Invoice"): return "#374151"
        if n.startswith("Vendor"): return "#111827"
        if n.startswith("Action"): return "#E5E7EB"
        if "[OK]" in n: return "#16A34A"
        if "[DUE]" in n: return "#F59E0B"
        if "[OVERDUE]" in n: return "#DC2626"
        if "[BLOCKED]" in n: return "#7C3AED"
        return "#9CA3AF"

    node_colors = [color_for(n) for n in G.nodes()]
    fig = plt.figure(figsize=(12,6))
    ax = fig.add_subplot(111)
    ax.axis("off")
    nx.draw_networkx_edges(G, pos, ax=ax, arrows=True, arrowstyle="-|>", arrowsize=10, width=1.0, alpha=0.6)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=1200, edgecolors="#111827", linewidths=1.0)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=8)
    st.pyplot(fig, clear_figure=True)
