# cars_ai

Replica MVP del progetto descritto in `../Can I Make a Better AI Than AI.md`: auto
che imparano a guidare con rete neurale e algoritmo genetico scritti da zero.
Nessuna libreria di machine learning: solo `numpy` per l'algebra lineare e
`pygame` per rendering e collisioni.

L'allenamento gira su **tre circuiti strutturalmente diversi**; la gara finale si
corre su un quarto circuito mai visto in allenamento.

## Struttura

```
src/
├── config.py           # dataclass frozen: piste, auto, rete, GA, simulazione
├── neural_network.py   # rete feedforward 8 → 14 → 14 → 2, genoma piatto
├── genetic.py          # selezione, crossover uniforme, mutazione, elitismo
├── track.py            # generazione procedurale dei circuiti + maschera
├── track_check.py      # validazione geometrica di un circuito
├── obstacles.py        # semafori e pedoni: funzionanti ma non più cablati
├── car.py              # fisica, 7 sensori raycast, freno, fitness
├── renderer.py         # finestra: circuito a sinistra, pannello a destra
├── dashboard.py        # pannello: stato, grafico fitness, rete del leader
├── simulation.py       # loop generazionale su piste multiple
└── cli.py              # argomenti e configurazione condivisi
train.py                # allena sui tre circuiti
race.py                 # porta il campione sul circuito mai visto
tests/
├── test_core.py        # rete, operatori genetici, pista, auto
├── test_obstacles.py   # modulo ostacoli mobili, in isolamento
└── test_tracks.py      # validità geometrica di tutti i circuiti
```

## Setup

Serve Python 3.12: su Python 3.14 il modulo `pygame.font` non è disponibile
(`ImportError: cannot import name 'Font'`) e il rendering non parte.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Uso

```bash
python train.py                                # allenamento con dashboard (ESC per uscire)
python train.py --headless --generations 200   # allenamento veloce, senza finestra
python race.py                                 # gara sul circuito mai allenato
python race.py --track 1                       # rivedi il campione su un circuito di training
python -m pytest tests -q                      # test
```

`race.py` usa `outputs/best_genome.npz` se non gli passi un percorso diverso.

## Come funziona

**Percezione.** Ogni auto proietta 7 raggi su un ventaglio di 180° attorno alla
propria direzione; ognuno avanza a passi di 3 px finché non esce dall'asfalto.
Alla rete arrivano 7 distanze normalizzate più la propria velocità: 8 input.
Gli isolotti sono ritagliati nella maschera della strada, quindi i raggi li
vedono come muri senza bisogno di alcun canale sensoriale dedicato.

**Cervello.** Rete densa `8 → 14 → 14 → 2` con `tanh` su ogni layer (366
parametri). Le uscite sono sterzo (−1…1) e acceleratore, dove un valore negativo
frena. I pesi non sono mai addestrati per gradiente: sono un genoma piatto che
l'algoritmo genetico ricombina. `genetic.py` non importa `neural_network.py` —
cambiare architettura non richiede modifiche agli operatori evolutivi.

**Fisica.** L'auto va solo in avanti. L'autorità di sterzo cresce con la velocità
fino a 1.5 px/frame, poi satura: da lì il raggio di curvatura è
`velocità / velocità_angolare`, cioè 76 px a velocità massima e 19 px a bassa
velocità. Frenare è ciò che permette di chiudere un tornante. Con un raggio
costante i tornanti stretti sarebbero impercorribili a qualunque velocità, non
difficili.

**Circuiti.** La centerline è una polilinea chiusa, smussata con l'algoritmo di
Chaikin e ricampionata a passo di 4 px. Quattro layout:

| Circuito | Uso | Giro | Problema che pone |
|---|---|---|---|
| serpentine | training | 3 670 px | tornanti a 180°, curve continue |
| grid-city | training | 3 107 px | angoli retti: frenare dentro la curva |
| speedway | training | 2 083 px | portare velocità, poi la chicane |
| gauntlet | **solo gara** | 2 366 px | zigzag continui, mai allenato |

I tornanti della serpentine sono archi semicircolari espliciti: lasciare uno
spigolo da smussare produce un raggio troppo stretto per essere guidato (14 px
invece di 75).

**Isolotti.** Quattro o cinque per circuito, alternati fra esagoni, triangoli,
quadrati e pentagoni. Sono dimensionati dalla larghezza locale della strada per
lasciare almeno 26 px di corridoio per lato, e dove non c'è spazio non vengono
disegnati: per costruzione non è possibile generare una pista tappata.

**Validazione geometrica** (`track_check.py`), eseguita dai test su ogni pista:
- due tratti che si toccano darebbero scorciatoie e falserebbero il progresso;
- una curva più stretta del raggio di sterzata è impercorribile;
- il corridoio percorribile più stretto deve restare guidabile;
- il layout non deve uscire dalla finestra.

Due vincoli scoperti così: le corsie della serpentine devono essere in **numero
pari** (con un numero dispari l'ultima finisce dal lato sbagliato e il tracciato
taglia sé stesso) e l'ampiezza delle gobbe deve stare sotto
`(distanza_corsie − larghezza_strada − 10) / 2`, altrimenti corsie adiacenti si
fondono.

**Fitness.** Distanza percorsa lungo la centerline, misurata proiettando l'auto
sul campione più vicino con ricerca in una finestra locale (costo costante per
frame). Il punteggio di ogni circuito è normalizzato in **giri**, non in pixel:
sommare pixel grezzi permetterebbe a un genoma di ignorare il circuito più
difficile e ottimizzare solo i due veloci.

**Evoluzione.** Il 30% migliore forma il breeding pool; i 4 migliori passano
intatti (elitismo), il resto nasce da crossover uniforme con mutazione gaussiana
sul 5% dei geni. Ogni genoma è valutato su tutti e tre i circuiti prima della
selezione.

## Dashboard

Finestra 1380×700: circuito a sinistra, pannello a destra con generazione,
circuito corrente (`TRACK k/3`), avanzamento, statistiche del run, grafico del
best per generazione e la rete del leader — connessioni verdi per pesi positivi,
rosse per negativi, intensità proporzionale al valore. Il diagramma è in cache e
si ridisegna solo quando cambia il leader.

## Ostacoli mobili (non attivi)

`obstacles.py` implementa semafori a tempo e pedoni che attraversano, con
raycast analitico su cerchi. Non sono più cablati nella simulazione, ma il modulo
resta coperto dai test: bastano `num_traffic_lights` / `num_pedestrians` in
`ObstacleConfig` per riattivarli.

Sono stati tolti per due ragioni misurate. I semafori facevano dominare il tempo
di attesa al rosso. I pedoni si sono rivelati non apprendibili da una rete senza
memoria: vedendo solo la distanza istantanea, non può distinguere un pedone che
le taglia la strada da uno fermo — su otto prove in gara, sette morti per
investimento e zero uscite di pista.

## Differenze rispetto al progetto originale

- Circuiti generati proceduralmente invece che disegnati in Photoshop, così il
  progetto è riproducibile senza asset esterni.
- Uscite continue (sterzo, acceleratore) invece di azioni discrete.
- Gli isolotti in carreggiata sono un'aggiunta: l'originale aveva solo i bordi.
- Non è implementata la gara multi-modello contro Claude/ChatGPT/Gemini: qui
  corre una sola popolazione.
