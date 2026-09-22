"""Create a local Leaflet review page from the candidate-path GeoJSON."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


def render_review_html(geojson: dict) -> str:
    if geojson.get("type") != "FeatureCollection":
        raise ValueError("Expected a GeoJSON FeatureCollection")
    roles = {feature.get("properties", {}).get("role") for feature in geojson.get("features", [])}
    if not roles.issubset({"preferred_observed", "rejected_suggestion", "context_road"}):
        raise ValueError("GeoJSON contains an unsupported path role")
    embedded = json.dumps(geojson, separators=(",", ":")).replace("</", "<\\/")
    title = html.escape("STGAT-LSTM Real Candidate Review")
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
  <style>
    html, body {{ height: 100%; margin: 0; font: 14px system-ui, sans-serif; }}
    body {{ display: grid; grid-template-columns: 320px 1fr; }}
    aside {{ padding: 20px; overflow: auto; border-right: 1px solid #d7dce2; background: #f7f8fa; }}
    #map {{ height: 100%; background: #eef1f4; }}
    select {{ width: 100%; padding: 8px; margin: 8px 0 16px; }}
    .legend {{ display: grid; gap: 8px; margin: 16px 0; }}
    .swatch {{ display: inline-block; width: 28px; height: 4px; margin-right: 8px; vertical-align: middle; }}
    .preferred {{ background: #138a36; }}
    .rejected {{ background: #c62828; }}
    .notice {{ padding: 10px; background: #fff3cd; border: 1px solid #ffe69c; line-height: 1.4; }}
    #details {{ line-height: 1.45; overflow-wrap: anywhere; }}
    @media (max-width: 760px) {{ body {{ grid-template-columns: 1fr; grid-template-rows: auto 65vh; }} aside {{ border-right: 0; }} }}
  </style>
</head>
<body>
  <aside>
    <h1>Candidate path review</h1>
    <div class="notice">These paths are review candidates. Viewing them does not approve them for model training.</div>
    <label for="event"><strong>Event</strong></label>
    <select id="event"></select>
    <div class="legend">
      <div><span class="swatch preferred"></span>Observed rider path</div>
      <div><span class="swatch rejected"></span>Prior suggested path</div>
    </div>
    <div id="details"></div>
  </aside>
  <div id="map"></div>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script>
    const collection = {embedded};
    const map = L.map('map');
    const select = document.getElementById('event');
    const details = document.getElementById('details');
    const groups = {{}};
    const contextFeatures = collection.features.filter(feature => feature.properties.role === 'context_road');
    if (contextFeatures.length) {{
      L.geoJSON({{type: 'FeatureCollection', features: contextFeatures}}, {{
        style: {{color: '#aeb7c2', weight: 1.2, opacity: 0.75}},
        interactive: false
      }}).addTo(map);
    }}
    collection.features.filter(feature => feature.properties.role !== 'context_road').forEach(feature => {{
      const eventKey = feature.properties.event_key;
      (groups[eventKey] ??= []).push(feature);
    }});
    Object.keys(groups).sort().forEach((eventKey, index) => {{
      const option = document.createElement('option');
      option.value = eventKey;
      option.textContent = `Candidate ${{index + 1}} (${{eventKey}})`;
      select.appendChild(option);
    }});
    let layer;
    function safe(value) {{
      const node = document.createElement('span');
      node.textContent = String(value ?? '');
      return node.innerHTML;
    }}
    function show(eventKey) {{
      if (layer) map.removeLayer(layer);
      const features = groups[eventKey];
      layer = L.geoJSON({{type: 'FeatureCollection', features}}, {{
        style: feature => feature.properties.role === 'preferred_observed'
          ? {{color: '#138a36', weight: 6, opacity: 0.85}}
          : {{color: '#c62828', weight: 5, opacity: 0.8, dashArray: '10 8'}},
        onEachFeature: (feature, item) => item.bindPopup(
          `<strong>${{safe(feature.properties.role)}}</strong><br>` +
          `${{safe(feature.properties.length_m)}} m<br>${{safe(feature.properties.road_names)}}`
        )
      }}).addTo(map);
      map.fitBounds(layer.getBounds(), {{padding: [25, 25]}});
      const preferred = features.find(feature => feature.properties.role === 'preferred_observed').properties;
      const rejected = features.find(feature => feature.properties.role === 'rejected_suggestion').properties;
      details.innerHTML = `<h2>${{safe(preferred.primary_reason)}}</h2>` +
        `<p><strong>Observed:</strong> ${{safe(preferred.length_m)}} m<br>${{safe(preferred.road_names)}}</p>` +
        `<p><strong>Suggested:</strong> ${{safe(rejected.length_m)}} m<br>${{safe(rejected.road_names)}}</p>`;
    }}
    select.addEventListener('change', () => show(select.value));
    if (select.options.length) show(select.value); else map.setView([14.57, 120.99], 13);
  </script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("geojson", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    geojson = json.loads(args.geojson.read_text(encoding="utf-8"))
    rendered = render_review_html(geojson)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(rendered, encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({"output": str(args.output), "features": len(geojson.get("features", []))}, indent=2))


if __name__ == "__main__":
    main()
