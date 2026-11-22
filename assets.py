"""
assets.py - original vegetable illustrations (inline SVG), food-themed CSS and
small HTML components for the GUI. No external images are used.
"""

BROCCOLI = """<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
<path d="M43 56 L39 92 Q50 97 61 92 L57 56 Z" fill="#9ccc65"/>
<path d="M46 66 L36 78" stroke="#8bc34a" stroke-width="6" stroke-linecap="round"/>
<circle cx="29" cy="46" r="16" fill="#2e7d32"/><circle cx="71" cy="46" r="16" fill="#2e7d32"/>
<circle cx="50" cy="31" r="20" fill="#388e3c"/><circle cx="39" cy="52" r="14" fill="#43a047"/>
<circle cx="61" cy="52" r="14" fill="#43a047"/>
<g fill="#1b5e20" opacity=".45"><circle cx="45" cy="26" r="3"/><circle cx="57" cy="34" r="3"/>
<circle cx="25" cy="44" r="3"/><circle cx="72" cy="40" r="3"/><circle cx="40" cy="50" r="2.5"/><circle cx="63" cy="54" r="2.5"/></g>
</svg>"""

CARROT = """<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg"><g transform="rotate(25 50 55)">
<path d="M50 30 Q65 30 62 43 L52 95 Q50 99 48 95 L38 43 Q35 30 50 30Z" fill="#f57c00"/>
<path d="M42 50 h8 M44 62 h7 M46 74 h5 M53 56 h7 M52 68 h5" stroke="#e65100" stroke-width="2.5" stroke-linecap="round"/>
<path d="M50 32 Q40 10 29 8 Q42 18 46 32Z" fill="#43a047"/><path d="M50 32 Q52 6 58 1 Q57 18 54 32Z" fill="#2e7d32"/>
<path d="M50 32 Q66 12 77 12 Q62 22 54 33Z" fill="#66bb6a"/></g></svg>"""

TOMATO = """<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
<circle cx="50" cy="59" r="34" fill="#e53935"/>
<ellipse cx="37" cy="47" rx="9" ry="5" fill="#fff" opacity=".35" transform="rotate(-30 37 47)"/>
<path d="M50 28 L56 18 L55 30 L67 26 L58 34 L65 42 L52 37 L50 46 L48 37 L35 42 L42 34 L33 26 L45 30 L44 18 Z" fill="#2e7d32"/>
<rect x="48" y="11" width="4" height="12" rx="2" fill="#33691e"/></svg>"""

EGGPLANT = """<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg"><g transform="rotate(-35 50 55)">
<ellipse cx="50" cy="63" rx="20" ry="32" fill="#6a1b9a"/><ellipse cx="43" cy="57" rx="4" ry="15" fill="#fff" opacity=".22"/>
<path d="M30 37 Q50 23 70 37 Q60 41 50 35 Q40 41 30 37Z" fill="#558b2f"/>
<rect x="47" y="15" width="6" height="16" rx="3" fill="#558b2f"/></g></svg>"""

LEAF = """<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
<path d="M50 92 C20 80 12 50 22 28 C30 12 46 8 50 8 C54 8 70 12 78 28 C88 50 80 80 50 92Z" fill="#7cb342"/>
<path d="M50 90 L50 18 M50 40 L34 28 M50 55 L30 44 M50 70 L34 62 M50 40 L66 28 M50 55 L70 44 M50 70 L66 62"
 stroke="#dcedc8" stroke-width="3" fill="none" stroke-linecap="round"/></svg>"""

PEPPER = """<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
<path d="M28 40 Q26 30 38 30 Q44 26 50 30 Q56 26 62 30 Q74 30 72 40 Q76 66 64 86 Q57 94 50 88 Q43 94 36 86 Q24 66 28 40Z" fill="#fbc02d"/>
<path d="M44 34 Q42 60 44 84" stroke="#f9a825" stroke-width="3" fill="none"/><path d="M57 34 Q59 60 56 84" stroke="#f9a825" stroke-width="3" fill="none"/>
<path d="M50 31 Q48 18 56 12" stroke="#33691e" stroke-width="5" fill="none" stroke-linecap="round"/></svg>"""

VEG = [BROCCOLI, CARROT, TOMATO, LEAF, EGGPLANT, PEPPER]


def icon(svg: str, size: int = 48) -> str:
    return f'<span class="veg" style="width:{size}px;height:{size}px">{svg}</span>'


CSS = """
<style>
:root { --leaf:#2e7d32; --leaf-2:#7cb342; --carrot:#f57c00; --tomato:#e53935; --cream:#fbfaf4; --ink:#1e2b1e; }
.veg { display:inline-block; vertical-align:middle; }
.veg svg { width:100%; height:100%; }
.hero { position:relative; overflow:hidden; border-radius:22px; padding:26px 30px; margin:0 0 14px 0;
  background: linear-gradient(120deg, #1b5e20 0%, #2e7d32 45%, #7cb342 100%); color:#fff;
  box-shadow: 0 10px 28px rgba(46,125,50,.25); }
.hero h1 { color:#fff !important; font-size:2.1rem; margin:0 0 4px 0; padding:0; letter-spacing:-.5px; }
.hero p { color:#f1f8e9; margin:0; font-size:1.02rem; }
.hero .chips { margin-top:12px; }
.hero .chip { display:inline-block; background:rgba(255,255,255,.18); border:1px solid rgba(255,255,255,.35);
  border-radius:999px; padding:3px 11px; margin:0 6px 6px 0; font-size:.82rem; color:#fff; }
.hero .basket { position:absolute; right:18px; top:50%; transform:translateY(-50%); display:flex; gap:4px; }
.hero .basket .veg { filter: drop-shadow(0 4px 6px rgba(0,0,0,.25)); }
@media (max-width: 1100px) { .hero .basket { display:none; } }
.veg-strip { display:flex; justify-content:space-between; padding:6px 4px 10px 4px; }
.esl { background:#fff; border:2px solid #263238; border-radius:14px; padding:14px 16px; max-width:420px;
  box-shadow: 0 6px 18px rgba(0,0,0,.08); font-family: ui-sans-serif, system-ui; }
.esl .top { display:flex; align-items:center; gap:12px; }
.esl .name { font-weight:700; font-size:1.15rem; color:var(--ink); }
.esl .sub { color:#5f6b5f; font-size:.85rem; }
.esl .price { font-size:2.6rem; font-weight:800; color:#1b5e20; line-height:1; margin:10px 0 2px 0; }
.esl .price small { font-size:1rem; font-weight:600; color:#5f6b5f; }
.esl .row { display:flex; justify-content:space-between; font-size:.88rem; color:#37474f; margin-top:6px; }
.esl .bar { height:9px; border-radius:6px; background:#e8eedf; overflow:hidden; margin-top:4px; }
.esl .bar > div { height:100%; border-radius:6px; }
.esl .foot { margin-top:10px; font-size:.72rem; color:#78909c; letter-spacing:.3px; }
div[data-testid="stMetric"] { background:#ffffff; border:1px solid #e1e8d5; border-radius:16px; padding:12px 14px;
  box-shadow: 0 2px 8px rgba(46,125,50,.06); }
div[data-testid="stMetric"] label p { color:#2e5e2e !important; font-weight:600; }
section[data-testid="stSidebar"] { border-right:1px solid #dfe8d2; }
button[data-baseweb="tab"] p { font-size:0.98rem; }
h2, h3 { color:#1f3d1f; }
.card { background:#fff; border:1px solid #e1e8d5; border-radius:16px; padding:14px 18px; margin-bottom:10px; }
.badge { display:inline-block; padding:2px 10px; border-radius:999px; font-size:.82rem; font-weight:600; margin:2px 4px 2px 0; }
.badge.ok { background:#e8f5e9; color:#1b5e20; border:1px solid #a5d6a7; }
.badge.warn { background:#fff3e0; color:#8a4b00; border:1px solid #ffcc80; }
.badge.info { background:#e3f2fd; color:#0d47a1; border:1px solid #90caf9; }
</style>
"""


def hero(day_txt: str) -> str:
    basket = "".join(icon(v, 64) for v in [BROCCOLI, CARROT, TOMATO, EGGPLANT, PEPPER])
    chips = "".join(f'<span class="chip">{c}</span>' for c in
                    ["Algorithm 1 inside", "Autonomous re-planning agent", "Cold-chain aware", "All-units discounts",
                     "Waste KPIs"])
    return f"""<div class="hero"><div class="basket">{basket}</div>
<h1>FreshTwin</h1><p>Digital twin for pricing and replenishment of perishable food</p>
<p style="opacity:.85;font-size:.9rem;margin-top:4px">{day_txt}</p><div class="chips">{chips}</div></div>"""


def veg_strip() -> str:
    return '<div class="veg-strip">' + "".join(icon(v, 40) for v in VEG) + "</div>"


def shelf_label(name, price, order, next_days, on_hand, waste_pct, age_days, shelf_life):
    if on_hand <= 1e-6:
        fresh_txt, fresh, fresh_col = "sold out", 0.0, "#b0bec5"
    else:
        fresh = max(0.0, 1 - (age_days or 0) / shelf_life) if shelf_life else 1.0
        fresh_txt = f"{100 * fresh:.0f}%"
        fresh_col = "#43a047" if fresh > 0.6 else ("#f9a825" if fresh > 0.3 else "#e53935")
    w = min(100.0, waste_pct if waste_pct == waste_pct else 0.0)
    return f"""<div class="esl"><div class="top">{icon(BROCCOLI, 54)}<div><div class="name">{name}</div>
<div class="sub">loose, per kg &middot; electronic shelf label</div></div></div>
<div class="price">{price:.2f} <small>CNY / kg</small></div>
<div class="row"><span>Order today</span><b>{order:.1f} kg</b></div>
<div class="row"><span>Next delivery in</span><b>{next_days} day(s)</b></div>
<div class="row"><span>On hand at close</span><b>{on_hand:.1f} kg</b></div>
<div class="row"><span>Freshness of oldest lot</span><b>{fresh_txt}</b></div>
<div class="bar"><div style="width:{100 * fresh:.0f}%;background:{fresh_col}"></div></div>
<div class="row"><span>Waste, last 28 days</span><b>{w:.1f}% of purchases</b></div>
<div class="bar"><div style="width:{min(100, 3 * w):.0f}%;background:#f57c00"></div></div>
<div class="foot">PRICE SET BY DIGITAL TWIN &middot; UPDATED EACH MORNING</div></div>"""
