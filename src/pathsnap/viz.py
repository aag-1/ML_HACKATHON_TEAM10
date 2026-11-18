"""Folium route maps: GPS observations, nearby road network, reference route, predicted routes (layer control)."""
import folium
import numpy as np


def _lines(nodes, arcs, ids):
    a, b = nodes[arcs[ids, 0]], nodes[arcs[ids, 1]]
    return [[[a[i, 1], a[i, 0]], [b[i, 1], b[i, 0]]] for i in range(len(ids))]


def route_map(trip, title, routes, gps_points=None, note=""):
    """routes: list of (layer name, arc id list, colour, weight). Reference route, if wanted, is passed by the caller."""
    lat, lon = trip.track.lat.to_numpy(), trip.track.lon.to_numpy()
    m = folium.Map(location=[float(np.median(lat)), float(np.median(lon))], tiles="OpenStreetMap", control_scale=True)
    m.fit_bounds([[lat.min(), lon.min()], [lat.max(), lon.max()]])

    pad, nd, A = 0.003, trip.nodes, trip.arcs
    a, b = nd[A[:, 0]], nd[A[:, 1]]
    near = np.where((np.minimum(a[:, 0], b[:, 0]) >= lon.min() - pad) & (np.maximum(a[:, 0], b[:, 0]) <= lon.max() + pad)
                    & (np.minimum(a[:, 1], b[:, 1]) >= lat.min() - pad) & (np.maximum(a[:, 1], b[:, 1]) <= lat.max() + pad))[0]
    net = folium.FeatureGroup(name="Road network", show=False)
    folium.PolyLine(_lines(nd, A, near), color="#9a9a9a", weight=1, opacity=.6).add_to(net)
    net.add_to(m)

    legend = ""
    for name, ids, colour, weight in routes:
        fg = folium.FeatureGroup(name=name, show=True)
        if len(ids):
            folium.PolyLine(_lines(nd, A, np.asarray(ids, int)), color=colour, weight=weight, opacity=.85).add_to(fg)
        fg.add_to(m)
        legend += f'<span style="color:{colour}">&#9473;&#9473;</span> {name}<br>'

    gps = folium.FeatureGroup(name="GPS observations", show=True)
    pts = gps_points if gps_points is not None else np.c_[lat, lon]
    step = max(1, len(pts) // 3000)
    for y, x in pts[::step]:
        folium.CircleMarker([y, x], radius=2, color="#1d4ed8", fill=True, fill_opacity=.8, weight=0).add_to(gps)
    gps.add_to(m)
    folium.Marker([lat[0], lon[0]], tooltip="start", icon=folium.Icon(color="green", icon="play")).add_to(m)
    folium.Marker([lat[-1], lon[-1]], tooltip="end", icon=folium.Icon(color="red", icon="stop")).add_to(m)
    html = (f'<div style="position:fixed;bottom:24px;left:24px;z-index:9999;background:white;padding:8px 10px;'
            f'border:1px solid #999;border-radius:6px;font:12px sans-serif"><b>{title}</b><br>{note}'
            f'<span style="color:#1d4ed8">&#9679;</span> GPS observation<br>{legend}'
            f'<span style="color:#9a9a9a">&#9473;&#9473;</span> road network (toggle)</div>')
    m.get_root().html.add_child(folium.Element(html))
    folium.LayerControl(collapsed=False).add_to(m)
    return m
