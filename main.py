from statsbombpy import sb
import matplotlib.pyplot as plt
from mplsoccer import Pitch
import pandas as pd
import openai
import json
from dotenv import load_dotenv
import os
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

# --- OpenAI Key ---
load_dotenv()  # carica le variabili dal .env

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("⚠️ Variabile OPENAI_API_KEY non trovata nel file .env.")




# =========================
# 1. Impostazioni e dati partita
# =========================
match_id = 22912
events = sb.events(match_id=match_id)


# =========================
# 2. Filtra i tiri
# =========================
shots = events[events['type'] == 'Shot']

def safe_get(series_or_dict, *keys, default=None):
    """Accesso sicuro a chiavi o colonne annidate."""
    val = series_or_dict
    for k in keys:
        if val is None:
            return default
        try:
            if isinstance(val, pd.Series) and k in val.index:
                val = val[k]
                continue
        except Exception:
            pass
        if isinstance(val, dict):
            val = val.get(k, default)
            continue
        if isinstance(val, (list, tuple)) and isinstance(k, int):
            if 0 <= k < len(val):
                val = val[k]
                continue
            else:
                return default
        try:
            val = val[k]
        except Exception:
            return default
    return val

# =========================
# 3. Estrai dati dei tiri
# =========================
x, y, shot_outcome, on_target, shot_team = [], [], [], [], []

team_list = []
for _, row in shots.iterrows():
    team_name = safe_get(row, 'team')
    if isinstance(team_name, dict):
        team_name = team_name.get('name')
    if team_name and team_name not in team_list:
        team_list.append(team_name)

if len(team_list) == 0:
    team_list = ['Team 1', 'Team 2']
elif len(team_list) == 1:
    team_list.append('Team 2')

color_map = {team_list[0]: 'red', team_list[1]: 'blue'}
goals = {team_list[0]: 0, team_list[1]: 0}

for _, row in shots.iterrows():
    loc = safe_get(row, 'location') or safe_get(row, 'shot', 'location')
    if not loc or not isinstance(loc, (list, tuple)) or len(loc) < 2:
        continue

    x_val, y_val = loc[0], loc[1]
    outcome = safe_get(row, 'shot_outcome') or safe_get(row, 'shot', 'outcome')
    if isinstance(outcome, dict):
        outcome = outcome.get('name') or outcome.get('type') or str(outcome)

    team = safe_get(row, 'team')
    if isinstance(team, dict):
        team = team.get('name')

    if team not in goals:
        goals[team] = 0

    if outcome and str(outcome).lower() == 'goal':
        goals[team] += 1

    x.append(x_val)
    y.append(y_val)
    shot_outcome.append(outcome)
    shot_team.append(team)

# =========================
# 4. Crea la shotmap
# =========================
pitch = Pitch(line_color='black')
fig, ax = pitch.draw(figsize=(10, 6))

markers = ['o' if (o and str(o).lower() == 'goal') else 'x' for o in shot_outcome]
colors = [color_map.get(t, 'grey') for t in shot_team]

for xi, yi, c, m in zip(x, y, colors, markers):
    ax.scatter(xi, yi, c=c, marker=m, s=120, edgecolors='black', alpha=0.85)

team1, team2 = team_list[0], team_list[1]
score1, score2 = goals.get(team1, 0), goals.get(team2, 0)
result_text = f"{team1} {score1} - {score2} {team2}"

ax.set_title(f"Shot Map – {result_text}", fontsize=14, fontweight='bold')

# =========================
# 5. Prepara i dati per GPT
# =========================
shots_data = []
for xi, yi, outcome, team in zip(x, y, shot_outcome, shot_team):
    shots_data.append({
        "x": xi,
        "y": yi,
        "outcome": outcome,
        "team": team
    })

# =========================
# 6. Prompt per GPT
# =========================
prompt = f"""
Sei un analista calcistico professionista. Ti fornisco i dati dei tiri di una partita (coordinate StatsBomb, campo 120x80):
- Match: {result_text}
- Numero totale di tiri: {len(shots_data)}

Ogni tiro contiene coordinate (x,y), squadra e risultato.

Ecco i dati:
{json.dumps(shots_data, indent=2)}

Scrivi un'analisi dettagliata che includa:
- Dove hanno tirato di più le squadre
- L'efficacia sotto porta (gol / tiri)
- Le differenze tra i due team in termini di posizionamento e pericolosità
- Considerazioni tattiche generali sui tiri
"""

# =========================
# 7. Chiamata al modello GPT-3.5-mini
# =========================
client = openai.OpenAI(api_key=api_key)  # ✅ Usa la chiave caricata in alto

response = client.chat.completions.create(
    model="gpt-4",    messages=[
        {"role": "system", "content": "Sei un esperto di analisi calcistica e tattica sportiva."},
        {"role": "user", "content": prompt}
    ],
    temperature=0.7
)

commento = response.choices[0].message.content
print("\n--- Analisi GPT-4 ---\n")


# =========================
# 8. Esporta in PDF con commento e immagine
# =========================


# Salva la figura della shotmap come immagine temporanea
plot_path = "shotmap.png"
fig.savefig(plot_path, dpi=300, bbox_inches='tight')

# Crea il documento PDF
pdf_path = "analisi_partita.pdf"
doc = SimpleDocTemplate(pdf_path, pagesize=A4)
styles = getSampleStyleSheet()

# Titolo e contenuti
elements = []
elements.append(Paragraph(f"<b>Shot Map & Analisi – {result_text}</b>", styles["Title"]))
elements.append(Spacer(1, 12))
elements.append(Image(plot_path, width=6*inch, height=3.5*inch))
elements.append(Spacer(1, 12))
elements.append(Paragraph("<b>Analisi GPT:</b>", styles["Heading2"]))
elements.append(Spacer(1, 8))
elements.append(Paragraph(commento.replace("\n", "<br/>"), styles["BodyText"]))

# Salva il file
doc.build(elements)
print(f"\n📄 PDF generato con successo: {pdf_path}")
