import plotly.graph_objects as go


def make_trend_chart(monthly_trend: dict) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=monthly_trend["months"],
            y=monthly_trend["이탈률"],
            mode="lines+markers",
            name="평균 예측 이탈률",
            line=dict(color="#FF0A16", width=4),
            marker=dict(size=7, color="#FF0A16"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=monthly_trend["months"],
            y=monthly_trend["활동도"],
            mode="lines+markers",
            name="평균 최근 시청 횟수",
            line=dict(color="#3B82F6", width=3),
            marker=dict(size=6, color="#3B82F6"),
            opacity=0.9,
            yaxis="y2",
        )
    )

    fig.update_layout(
        height=380,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.05,
            xanchor="left",
            x=0,
            font=dict(color="#E5E7EB"),
        ),
        xaxis=dict(
            showgrid=False,
            tickfont=dict(color="#D1D5DB"),
        ),
        yaxis=dict(
            title=dict(
                text="예측 이탈률(%)",
                font=dict(color="#FFB4B4"),
            ),
            showgrid=True,
            gridcolor="rgba(255,255,255,0.08)",
            zeroline=False,
            tickfont=dict(color="#D1D5DB"),
        ),
        yaxis2=dict(
            title=dict(
                text="최근 시청 횟수",
                font=dict(color="#A5C8FF"),
            ),
            overlaying="y",
            side="right",
            showgrid=False,
            zeroline=False,
            tickfont=dict(color="#D1D5DB"),
        ),
    )
    return fig


def make_risk_donut(risk_segments: dict) -> go.Figure:
    colors = ["#E50914", "#F97316", "#FACC15", "#22C55E"]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=risk_segments["labels"],
                values=risk_segments["values"],
                hole=0.65,
                marker=dict(colors=colors),
                textinfo="none",
                sort=False,
            )
        ]
    )

    fig.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        annotations=[
            dict(
                text=f"<b>{sum(risk_segments['values']):,}</b><br>총 사용자",
                x=0.5,
                y=0.5,
                font=dict(size=18, color="#F9FAFB"),
                showarrow=False,
            )
        ],
    )
    return fig


def make_genre_donut(items: list[dict]) -> go.Figure:
    labels = [g["label"] for g in items]
    values = [g["value"] for g in items]
    colors = ["#3B82F6", "#10B981", "#8B5CF6", "#F59E0B", "#EF4444"]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.65,
                marker=dict(colors=colors),
                textinfo="none",
                sort=False,
            )
        ]
    )

    fig.update_layout(
        height=280,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    return fig