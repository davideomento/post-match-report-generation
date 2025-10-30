from statsbombpy import sb
import matplotlib.pyplot as plt
from mplsoccer import Pitch
import openai

# =========================
# CONFIGURAZIONE
# =========================
MATCH_ID = 7478      # Inserisci la partita che vuoi
API_KEY = "YOUR_OPENAI_API_KEY"  # Inserisci la tua API key OpenAI

# =========================
# 1. Scarico eventi con statsbombpy
# =========================
events = sb.events(match_id=MATCH_ID)

# =========================
# 2. Filtra i tiri
# =========================
shots = events[events['type_name'] == 'Shot']

# Colori per squadra
teams = shots['team_name'].unique()
color_map = {teams[0]: 'red', teams[1]: 'blue'}

# =========================
# 3. Crea shotmap
# =========================
pitch = Pitch(line_color='black')
fig, ax = pitch.draw(figsize=(10, 6))

# Scatter con colore per squadra
colors = [color_map[t] for t in shots['team_name']]
pitch.scatter(shots['x'], shots['y'], ax=ax, c=colors, s=100, edgecolors='black', alpha=0.7)
plt.title("Shot Map - Colori per squadra")

# =========================
# 4. Prepara prompt per LLM
# =========================
# Creiamo lista con dati di ogni tiro
shot_data = [
    {
        'team': shots.iloc[i]['team_name'],
        'x': shots.iloc[i]['x'],
        'y': shots.iloc[i]['y'],
        'outcome': shots.iloc[i]['shot_outcome_name'],
        'on_target': shots.iloc[i]['shot']['end_location'] is not None
    }
    for i in range(len(shots))
]

prompt = f"""
Sei un esperto in match analysis. Analizza la seguente shot map della partita.
Ogni tiro è indicato con:
- squadra che ha tirato
- coordinate (x, y)
- outcome: Goal o No Goal
- se il tiro è stato in porta (on target) o fuori (off target)

Dati tiri:
{shot_data}

Fornisci un'analisi dettagliata dei tiri, pattern di attacco di entrambe le squadre, e osservazioni tattiche.
"""

# =========================
# 5. Genera descrizione con LLM
# =========================
openai.api_key = API_KEY

response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[{"role": "user", "content": prompt}]
)
llm_text = response['choices'][0]['message']['content']

# =========================
# 6. Mostra grafico e testo
# =========================
plt.show()
print("\nLLM Output:\n")
print(llm_text)
