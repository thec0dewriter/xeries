"""HTML report generation for dashboard outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from xeries.dashboard.results import DashboardResult


def render_html_report(results: DashboardResult, title: str = "TimeLens Dashboard Report") -> str:
    """Render a dashboard report to HTML using Jinja2."""
    try:
        from jinja2 import Template
    except ImportError as e:
        raise ImportError(
            "jinja2 is required for report generation. Install with: pip install jinja2"
        ) from e

    template = Template(
        """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{{ title }}</title>
  <style>
    body { font-family: Georgia, 'Times New Roman', serif; margin: 2rem; background: linear-gradient(120deg, #f8fbff, #eef4ff); color: #142033; }
    h1, h2 { margin-bottom: 0.3rem; }
    table { border-collapse: collapse; width: 100%; margin-bottom: 1.2rem; }
    th, td { border: 1px solid #c9d7eb; padding: 0.45rem 0.6rem; text-align: left; }
    th { background: #dfe9f7; }
    .section { background: #ffffffcc; border: 1px solid #d7e2f2; border-radius: 8px; padding: 1rem; margin-bottom: 1rem; }
  </style>
</head>
<body>
  <h1>{{ title }}</h1>
  <div class="section">
    <h2>Summary</h2>
    {{ summary_html | safe }}
  </div>
  {% if ranking_html %}
  <div class="section">
    <h2>Ranking Agreement</h2>
    {{ ranking_html | safe }}
  </div>
  {% endif %}
</body>
</html>
        """
    )

    summary_df = results.summary()
    ranking_df = results.compare_rankings()

    summary_html = summary_df.to_html(index=False) if not summary_df.empty else "<p>No data.</p>"
    ranking_html = ranking_df.to_html() if not ranking_df.empty else ""

    return template.render(title=title, summary_html=summary_html, ranking_html=ranking_html)


def write_html_report(path: str | Path, results: DashboardResult, title: str) -> Path:
    """Write dashboard report to disk."""
    output_path = Path(path)
    html = render_html_report(results=results, title=title)
    output_path.write_text(html, encoding="utf-8")
    return output_path


def show_html_report(results: DashboardResult, title: str = "TimeLens Dashboard Report") -> Path:
    """Write and open a temporary report in the default browser."""
    import tempfile
    import webbrowser

    temp_path = Path(tempfile.gettempdir()) / "timelens_dashboard_report.html"
    write_html_report(temp_path, results, title=title)
    webbrowser.open(temp_path.as_uri())
    return temp_path
