import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import re

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="TicketMind AI", page_icon="⚙️", layout="wide")

# ─── CUSTOM CSS ───
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Inter:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0a0a0f;
    color: #e2e8f0;
}

#MainMenu, footer, header {visibility: hidden;}
.block-container {padding: 2rem 3rem;}

[data-testid="stSidebar"] {
    background: #0d0d14;
    border-right: 1px solid #1e2030;
}

div[data-baseweb="select"] > div {
    border-color: #1e2030 !important;
    background-color: #0d0d14 !important;
}
div[data-baseweb="select"] > div:focus-within {
    border-color: #2a2a3a !important;
    box-shadow: none !important;
}

.sidebar-logo {
    padding: 1.5rem 1rem;
    border-bottom: 1px solid #1e2030;
    margin-bottom: 1.5rem;
}
.sidebar-logo h1 {
    font-family: 'JetBrains Mono', monospace;
    font-size: 18px;
    font-weight: 700;
    color: #00d4ff;
    margin: 0;
    letter-spacing: -0.02em;
}
.sidebar-logo p {
    font-size: 11px;
    color: #475569;
    margin: 4px 0 0 0;
    letter-spacing: 0.05em;
}

.page-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 28px;
    font-weight: 700;
    color: #f1f5f9;
    letter-spacing: -0.03em;
    margin-bottom: 0.25rem;
}
.page-subtitle {
    font-size: 13px;
    color: #475569;
    margin-bottom: 2rem;
    letter-spacing: 0.02em;
}

.metric-card {
    background: #0d0d14;
    border: 1px solid #1e2030;
    border-radius: 8px;
    padding: 1.25rem 1.5rem;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #00d4ff, #0066ff);
}
.metric-label {
    font-size: 11px;
    color: #475569;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.metric-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 32px;
    font-weight: 700;
    color: #f1f5f9;
    line-height: 1;
}
.metric-value.accent { color: #00d4ff; }
.metric-value.warning { color: #f59e0b; }
.metric-value.danger { color: #ef4444; }
.metric-value.success { color: #10b981; }

.section-header {
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    font-weight: 600;
    color: #00d4ff;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin: 2rem 0 1rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #1e2030;
}

.ticket-card {
    background: #0d0d14;
    border: 1px solid #1e2030;
    border-radius: 8px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 0.75rem;
    transition: border-color 0.2s;
}
.ticket-card:hover { border-color: #00d4ff40; }
.ticket-card-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 0.75rem;
}
.ticket-title {
    font-size: 15px;
    font-weight: 600;
    color: #f1f5f9;
}
.ticket-meta {
    display: flex;
    gap: 0.5rem;
    align-items: center;
    flex-wrap: wrap;
}

.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.badge-critical { background: #ef444420; color: #ef4444; border: 1px solid #ef444440; }
.badge-high { background: #f59e0b20; color: #f59e0b; border: 1px solid #f59e0b40; }
.badge-medium { background: #3b82f620; color: #3b82f6; border: 1px solid #3b82f640; }
.badge-low { background: #10b98120; color: #10b981; border: 1px solid #10b98140; }
.badge-escalate { background: #f59e0b20; color: #f59e0b; border: 1px solid #f59e0b40; }
.badge-auto_resolve { background: #10b98120; color: #10b981; border: 1px solid #10b98140; }

.confidence-bar-wrapper { margin-top: 0.5rem; }
.confidence-label { font-size: 11px; color: #475569; margin-bottom: 3px; }
.confidence-bar { height: 4px; background: #1e2030; border-radius: 2px; overflow: hidden; }
.confidence-fill { height: 100%; border-radius: 2px; }

.stTextInput input, .stTextArea textarea {
    background: #111118 !important;
    border: 1px solid #1e2030 !important;
    border-radius: 6px !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: #00d4ff !important;
    box-shadow: 0 0 0 2px #00d4ff20 !important;
}

.stButton button {
    background: linear-gradient(135deg, #00d4ff, #0066ff) !important;
    color: #0a0a0f !important;
    border: none !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    letter-spacing: 0.02em !important;
    padding: 0.5rem 1.5rem !important;
    transition: opacity 0.2s !important;
}
.stButton button:hover { opacity: 0.85 !important; }

.ai-analysis {
    background: #00d4ff08;
    border: 1px solid #00d4ff20;
    border-radius: 6px;
    padding: 1rem 1.25rem;
    font-size: 13px;
    line-height: 1.7;
    color: #94a3b8;
}

.result-card {
    background: #0d0d14;
    border: 1px solid #1e2030;
    border-radius: 8px;
    padding: 1.5rem;
    margin-top: 1.5rem;
}
.result-card.success { border-left: 3px solid #10b981; }
.result-card.warning { border-left: 3px solid #f59e0b; }

[data-testid="stExpander"] {
    background: #0d0d14 !important;
    border: 1px solid #1e2030 !important;
    border-radius: 8px !important;
    margin-bottom: 0.75rem !important;
}

.audit-row {
    display: flex;
    gap: 1rem;
    padding: 0.75rem 0;
    border-bottom: 1px solid #1e2030;
    align-items: flex-start;
}
.audit-agent {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: #00d4ff;
    min-width: 160px;
}
.audit-action {
    font-size: 11px;
    font-weight: 600;
    min-width: 140px;
}
.audit-reasoning {
    color: #64748b;
    flex: 1;
    font-size: 12px;
}
.audit-conf {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: #334155;
    min-width: 40px;
    text-align: right;
}
.audit-time {
    color: #334155;
    font-size: 11px;
    font-family: 'JetBrains Mono', monospace;
    min-width: 130px;
    text-align: right;
}

hr { border-color: #1e2030 !important; }
</style>
""", unsafe_allow_html=True)

# ─── SIDEBAR ───
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <h1><span style="color:#475569;font-size:14px;margin-right:6px;">TM</span>TicketMind</h1>
        <p>AI-POWERED SUPPORT SYSTEM</p>
    </div>
    """, unsafe_allow_html=True)
    page = st.selectbox("", ["Dashboard", "Submit Ticket", "Human Queue", "Audit Trail","Load Dataset"])

# ─── HELPERS ───
def priority_badge(p):
    p = (p or 'medium').lower()
    return f'<span class="badge badge-{p}">{p}</span>'

def action_badge(a):
    a = (a or '').lower()
    label = "Auto Resolved" if a == "auto_resolve" else "Escalated" if a == "escalate" else a
    return f'<span class="badge badge-{a}">{label}</span>'

def confidence_bar(score):
    pct = int((score or 0) * 100)
    color = "#10b981" if pct >= 75 else "#f59e0b" if pct >= 50 else "#ef4444"
    return f"""
    <div class="confidence-bar-wrapper">
        <div class="confidence-label">Confidence: {pct}%</div>
        <div class="confidence-bar">
            <div class="confidence-fill" style="width:{pct}%;background:{color};"></div>
        </div>
    </div>"""

def plotly_bar(labels, values, title, color="#00d4ff"):
    short_labels = [l[:22] + '...' if len(l) > 22 else l for l in labels]
    fig = go.Figure(go.Bar(
        x=short_labels,
        y=values,
        marker=dict(color=color, opacity=0.85),
        hovertemplate='<b>%{x}</b><br>Count: %{y}<extra></extra>'
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(family='JetBrains Mono', size=13, color='#00d4ff'), x=0),
        paper_bgcolor='#0d0d14',
        plot_bgcolor='#0d0d14',
        font=dict(family='Inter', color='#64748b', size=11),
        margin=dict(l=0, r=0, t=40, b=80),
        height=280,
        xaxis=dict(
            tickfont=dict(size=11, color='#475569'),
            gridcolor='#1e2030',
            linecolor='#1e2030',
            tickangle=-45,
            automargin=True
        ),
        yaxis=dict(
            tickfont=dict(size=11, color='#475569'),
            gridcolor='#1e2030',
            linecolor='#1e2030',
        ),
        hoverlabel=dict(bgcolor='#1e2030', font_size=12, font_family='Inter')
    )
    return fig

# ─── PAGE 1: DASHBOARD ───
if page == "Dashboard":
    st.markdown('<div class="page-title">Analytics Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Real-time overview of ticket processing pipeline</div>', unsafe_allow_html=True)

    try:
        stats = requests.get(f"{API_URL}/stats").json()
        tickets = requests.get(f"{API_URL}/tickets").json()
        df = pd.DataFrame(tickets)

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Total Tickets</div><div class="metric-value accent">{stats["total"]}</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Auto Resolved</div><div class="metric-value success">{stats["auto_resolved"]}</div></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Escalated</div><div class="metric-value warning">{stats["escalated"]}</div></div>', unsafe_allow_html=True)
        with col4:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Pending Review</div><div class="metric-value danger">{stats["pending_review"]}</div></div>', unsafe_allow_html=True)
        with col5:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Resolution Rate</div><div class="metric-value accent">{stats["resolution_rate"]}%</div></div>', unsafe_allow_html=True)

        if len(df) > 0:
            st.markdown('<div style="height:1.5rem"></div>', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                cat_counts = df['category'].value_counts()
                fig = plotly_bar(
                    cat_counts.index.tolist(),
                    cat_counts.values.tolist(),
                    "TICKETS BY CATEGORY",
                    "#00d4ff"
                )
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            with col2:
                action_counts = df['action'].value_counts()
                action_labels = ["Auto Resolved" if a == "auto_resolve" else "Escalated" if a == "escalate" else a for a in action_counts.index.tolist()]
                colors = ["#10b981" if a == "auto_resolve" else "#f59e0b" for a in action_counts.index.tolist()]
                fig2 = go.Figure(go.Bar(
                    x=action_labels,
                    y=action_counts.values.tolist(),
                    marker=dict(color=colors, opacity=0.85),
                    hovertemplate='<b>%{x}</b><br>Count: %{y}<extra></extra>'
                ))
                fig2.update_layout(
                    title=dict(text="ACTION BREAKDOWN", font=dict(family='JetBrains Mono', size=13, color='#00d4ff'), x=0),
                    paper_bgcolor='#0d0d14',
                    plot_bgcolor='#0d0d14',
                    font=dict(family='Inter', color='#64748b', size=11),
                    margin=dict(l=0, r=0, t=40, b=80),
                    height=280,
                    xaxis=dict(tickfont=dict(size=12, color='#94a3b8'), gridcolor='#1e2030', linecolor='#1e2030', automargin=True),
                    yaxis=dict(tickfont=dict(size=11, color='#475569'), gridcolor='#1e2030', linecolor='#1e2030'),
                    hoverlabel=dict(bgcolor='#1e2030', font_size=12)
                )
                st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar': False})

            st.markdown('<div class="section-header">Recent Tickets</div>', unsafe_allow_html=True)
            for _, t in df.head(8).iterrows():
                st.markdown(f"""
                <div class="ticket-card">
                    <div class="ticket-card-header">
                        <div class="ticket-title">{t.get('title','')}</div>
                        <div class="ticket-meta">
                            {priority_badge(t.get('priority',''))}
                            {action_badge(t.get('action',''))}
                        </div>
                    </div>
                    <div style="font-size:12px;color:#475569;margin-bottom:0.5rem;">
                        {t.get('category','')} · {str(t.get('created_at',''))[:16]}
                    </div>
                    {confidence_bar(t.get('confidence_score', 0))}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No tickets yet. Submit one to get started.")

    except Exception as e:
        st.error(f"Cannot connect to API. Make sure FastAPI is running. Error: {e}")

# ─── PAGE 2: SUBMIT TICKET ───
elif page == "Submit Ticket":
    st.markdown('<div class="page-title">Submit New Ticket</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">AI pipeline will classify and route automatically</div>', unsafe_allow_html=True)

    with st.form("ticket_form"):
        title = st.text_input("Ticket Title", placeholder="e.g. Cannot connect to VPN")
        description = st.text_area("Description", placeholder="Describe the issue in detail...", height=150)
        priority = st.selectbox("Priority", ["low", "medium", "high", "critical"])
        submitted = st.form_submit_button("→ Submit Ticket")

    if submitted and title and description:
        with st.spinner("Processing through AI pipeline..."):
            try:
                response = requests.post(f"{API_URL}/ticket", json={
                    "title": title,
                    "description": description,
                    "priority": priority
                })
                result = response.json()

                if result.get('action') == 'auto_resolve':
                    st.markdown(f"""
                    <div class="result-card success">
                        <div style="font-family:'JetBrains Mono',monospace;font-size:13px;color:#10b981;margin-bottom:0.75rem;">
                            AUTO-RESOLVED — {result['confidence_score']:.0%} confidence
                        </div>
                        <div style="font-size:12px;color:#475569;margin-bottom:0.75rem;">
                            Category: {result.get('category','')} &nbsp;·&nbsp; {priority_badge(priority)}
                        </div>
                        <div class="ai-analysis">{result.get('suggested_resolution','')}</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-card warning">
                        <div style="font-family:'JetBrains Mono',monospace;font-size:13px;color:#f59e0b;margin-bottom:0.75rem;">
                            ESCALATED — {result['confidence_score']:.0%} confidence · Routed to human review
                        </div>
                        <div style="font-size:12px;color:#475569;margin-bottom:0.75rem;">
                            Category: {result.get('category','')} &nbsp;·&nbsp; {priority_badge(priority)}
                        </div>
                        <div class="ai-analysis">{result.get('suggested_resolution', 'Awaiting human review')}</div>
                    </div>
                    """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error: {e}")

# ─── PAGE 3: HUMAN QUEUE ───
elif page == "Human Queue":
    st.markdown('<div class="page-title">Human Review Queue</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Tickets requiring human oversight and resolution</div>', unsafe_allow_html=True)

    try:
        tickets = requests.get(f"{API_URL}/tickets/escalated").json()

        if not tickets:
            st.markdown("""
            <div style="background:#10b98110;border:1px solid #10b98130;border-radius:8px;
                        padding:1rem 1.5rem;font-size:13px;color:#10b981;
                        font-family:'JetBrains Mono',monospace;">
                // Queue is clear — no tickets pending review
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="font-family:'JetBrains Mono',monospace;font-size:13px;
                        color:#f59e0b;margin-bottom:1.5rem;">
                [ {len(tickets)} ] tickets awaiting review
            </div>
            """, unsafe_allow_html=True)

            for ticket in tickets:
                conf = int((ticket.get('confidence_score') or 0) * 100)
                with st.expander(f"{ticket['title']}  ·  {(ticket.get('priority') or '').upper()}  ·  {conf}% confidence"):
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        st.markdown("**Ticket Details**")
                        st.markdown(f"""
                        <div style="font-size:13px;line-height:2.2;">
                            <div><span style="color:#475569">Category:</span>&nbsp;&nbsp;{ticket.get('category','')}</div>
                            <div><span style="color:#475569">Priority:</span>&nbsp;&nbsp;{priority_badge(ticket.get('priority',''))}</div>
                            <div><span style="color:#475569">Created:</span>&nbsp;&nbsp;<span style="font-family:'JetBrains Mono',monospace;font-size:11px;color:#64748b;">{str(ticket.get('created_at',''))[:16]}</span></div>
                        </div>
                        <div style="margin-top:1rem;font-size:13px;color:#94a3b8;line-height:1.7;
                                    background:#111118;border-radius:6px;padding:0.75rem 1rem;">
                            {ticket.get('description','')}
                        </div>
                        """, unsafe_allow_html=True)
                    with col2:
                        st.markdown("**AI Analysis**")
                        st.markdown(f'<div class="ai-analysis">{ticket.get("suggested_resolution","Awaiting analysis...")}</div>', unsafe_allow_html=True)

                    st.markdown('<div style="height:1rem"></div>', unsafe_allow_html=True)
                    human_resolution = st.text_area(
                        "Your Resolution",
                        value=ticket.get('suggested_resolution', ''),
                        key=f"res_{ticket['id']}",
                        height=120
                    )

                    col1, col2, col3 = st.columns([2, 2, 6])
                    with col1:
                        if st.button("Approve & Resolve", key=f"approve_{ticket['id']}"):
                            response = requests.post(
                                f"{API_URL}/tickets/{ticket['id']}/resolve",
                                json={"resolution": human_resolution}
                            )
                            if response.status_code == 200:
                                st.success("Resolved and learned!")
                                st.rerun()
                            else:
                                st.error("Failed to resolve.")
                    with col2:
                        if st.button("Reject", key=f"reject_{ticket['id']}"):
                            requests.post(
                                f"{API_URL}/tickets/{ticket['id']}/resolve",
                                json={"resolution": "REJECTED"}
                            )
                            st.warning("Ticket rejected.")
                            st.rerun()

    except Exception as e:
        st.error(f"Cannot connect to API: {e}")

# ─── PAGE 4: AUDIT TRAIL ───
elif page == "Audit Trail":
    st.markdown('<div class="page-title">Audit Trail</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Every agent decision logged for full explainability</div>', unsafe_allow_html=True)

    try:
        logs = requests.get(f"{API_URL}/audit").json()

        if not logs:
            st.info("No audit logs yet.")
        else:
            st.markdown('<div class="section-header">Agent Activity Log</div>', unsafe_allow_html=True)
            for log in logs:
                action = log.get('action', '')
                # Clean any markdown formatting from action text
                action_clean = re.sub(r'\*\*.*?\*\*', '', action).strip()
                action_clean = re.sub(r'\*', '', action_clean).strip()
                color = "#10b981" if "resolved" in action or "learned" in action else "#f59e0b" if "escalat" in action else "#00d4ff"
                # Clean reasoning too
                reasoning = re.sub(r'\*\*.*?\*\*', '', log.get('reasoning', '')).strip()
                reasoning = re.sub(r'\*', '', reasoning).strip()
                st.markdown(f"""
                <div class="audit-row">
                    <div class="audit-agent">{log.get('agent','')}</div>
                    <div class="audit-action" style="color:{color};">{action_clean}</div>
                    <div class="audit-reasoning">{reasoning}</div>
                    <div class="audit-conf">{str(log.get('confidence') or 0)[:4]}</div>
                    <div class="audit-time">{str(log.get('timestamp',''))[:16]}</div>
                </div>
                """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Cannot connect to API: {e}")


# ─── PAGE 5: LOAD DATASET ───
elif page == "Load Dataset":
    st.markdown('<div class="page-title">Load Dataset</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Upload any IT support CSV — AI will map columns automatically</div>', unsafe_allow_html=True)

    # Info card
    st.markdown("""
    <div style="background:#00d4ff08;border:1px solid #00d4ff20;border-radius:8px;
                padding:1.25rem 1.5rem;margin-bottom:2rem;">
        <div style="font-family:'JetBrains Mono',monospace;font-size:12px;color:#00d4ff;
                    margin-bottom:0.75rem;">HOW IT WORKS</div>
        <div style="font-size:13px;color:#64748b;line-height:2;">
            1. Upload any CSV from ServiceNow, Zendesk, Jira, or custom systems<br>
            2. AI detects which columns map to ticket subject, description, and resolution<br>
            3. All tickets are embedded and added to the knowledge base<br>
            4. System immediately uses new data to resolve future tickets
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded_file = st.file_uploader(
            "Upload CSV file",
            type=['csv'],
            help="Any CSV with ticket data — columns will be auto-detected"
        )
    with col2:
        source_name = st.text_input(
            "Dataset Name",
            value="custom_dataset",
            help="Label for this dataset in the knowledge base"
        )

    if uploaded_file is not None:
        # Preview the file
        try:
            import io
            df_preview = pd.read_csv(uploaded_file)
            uploaded_file.seek(0)  # Reset for upload

            st.markdown('<div class="section-header">File Preview</div>', unsafe_allow_html=True)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Total Rows</div>
                    <div class="metric-value accent">{len(df_preview)}</div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Columns</div>
                    <div class="metric-value accent">{len(df_preview.columns)}</div>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">File Size</div>
                    <div class="metric-value accent">{uploaded_file.size // 1024}KB</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown('<div style="height:1rem"></div>', unsafe_allow_html=True)

            # Show columns detected
            st.markdown("""
            <div style="font-family:'JetBrains Mono',monospace;font-size:12px;
                        color:#475569;margin-bottom:0.5rem;">COLUMNS DETECTED</div>
            """, unsafe_allow_html=True)
            cols_html = ''.join([f'<span class="badge badge-medium" style="margin:2px;">{c}</span>' for c in df_preview.columns])
            st.markdown(f'<div style="margin-bottom:1rem;">{cols_html}</div>', unsafe_allow_html=True)

            # Show sample data
            st.dataframe(df_preview.head(5), use_container_width=True)

            st.markdown('<div style="height:1rem"></div>', unsafe_allow_html=True)

            if st.button("→ Load into Knowledge Base"):
                with st.spinner("AI mapping columns and embedding tickets..."):
                    try:
                        response = requests.post(
                            f"{API_URL}/upload-dataset",
                            files={"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")},
                            data={"source_name": source_name}
                        )
                        result = response.json()

                        if response.status_code == 200 and result.get('success'):
                            st.markdown(f"""
                            <div class="result-card success">
                                <div style="font-family:'JetBrains Mono',monospace;font-size:13px;
                                            color:#10b981;margin-bottom:1rem;">
                                    DATASET LOADED SUCCESSFULLY
                                </div>
                                <div style="font-size:13px;color:#64748b;line-height:2.2;">
                                    <div>Filename: <span style="color:#e2e8f0;">{result['filename']}</span></div>
                                    <div>Rows processed: <span style="color:#e2e8f0;">{result['processed']}</span></div>
                                    <div>Rows skipped: <span style="color:#475569;">{result['skipped']}</span></div>
                                    <div>Tickets added to KB: <span style="color:#10b981;font-weight:600;">{result['tickets_added']}</span></div>
                                    <div>KB size: <span style="color:#475569;">{result['kb_before']}</span> → <span style="color:#00d4ff;font-weight:600;">{result['kb_after']}</span></div>
                                </div>
                                <div style="margin-top:1rem;font-family:'JetBrains Mono',monospace;
                                            font-size:12px;color:#475569;">
                                    COLUMN MAPPING DETECTED<br>
                                    <span style="color:#94a3b8;">
                                        subject → {result['mapping_detected'].get('subject','')}<br>
                                        body → {result['mapping_detected'].get('body','')}<br>
                                        answer → {result['mapping_detected'].get('answer','')}
                                    </span>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.error(f"Failed: {result.get('detail', 'Unknown error')}")

                    except Exception as e:
                        st.error(f"Error: {e}")

        except Exception as e:
            st.error(f"Could not read CSV: {e}")