# DigitalTwinSim

## Simulazione Predittiva Multi-Agente per il Decision-Making Legislativo

---

**Alberto Giovanni Gerli**
Tourbillon Tech srl

**Per Walter Rizzetto**
Presidente XI Commissione Lavoro — Camera dei Deputati

Marzo 2026

---

## Walter,

ti scrivo per mostrarti qualcosa su cui lavoro da mesi e che credo possa essere uno strumento concreto per il lavoro che fai in Commissione.

Sai che vengo dal mondo dei dati e della modellistica predittiva — dal COVID in poi ho continuato a sviluppare sistemi sempre piu sofisticati. DigitalTwinSim e il risultato: una piattaforma che costruisce un **gemello digitale della societa italiana** attorno a qualsiasi scenario, e simula cosa succederebbe se una certa decisione venisse presa.

Non e un sondaggio. Non e un chatbot che ti da un'opinione. E un motore di simulazione serio, con un'architettura ingegneristica complessa alle spalle, che riproduce le dinamiche reali di opinione, comunicazione e polarizzazione.

Te lo spiego in modo che tu possa capire la profondita tecnica senza doverti leggere il codice.

---

## Il Motore: Cosa c'e Dentro

### Architettura a Tre Livelli di Agenti

Il cuore del sistema e una simulazione **multi-agente a tre livelli gerarchici**, dove ogni livello ha un proprio modello comportamentale, un proprio grado di autonomia e un proprio peso nel determinare le dinamiche complessive:

**Livello 1 — Agenti Elite (Tier 1)**
Figure pubbliche reali: politici, leader sindacali, CEO, editorialisti, economisti. Ogni agente ha un profilo completo: posizione iniziale calibrata su dichiarazioni pubbliche verificate, stile comunicativo estratto da analisi del linguaggio reale, grado di influenza pesato per reach mediatico, rigidita ideologica (quanto e disposto a spostarsi), rete di alleanze e antagonismi. Questi agenti non "parlano a caso" — generano dichiarazioni coerenti con il loro profilo, reagiscono a cio che gli altri agenti hanno detto nei round precedenti, e possono formare coalizioni o romperle.

**Livello 2 — Agenti Istituzionali (Tier 2)**
Enti come INPS, Banca d'Italia, ISTAT, Confindustria, CGIL, Commissione Europea. Hanno logiche di risposta diverse dagli individui: sono piu lenti, piu formali, piu vincolati. Un'istituzione non cambia posizione da un giorno all'altro — il modello rispetta questa inerzia. Ma quando si muove, il suo peso sull'opinione pubblica e massivo.

**Livello 3 — Cluster Demografici (migliaia di agenti sintetici)**
La popolazione. Non un "pubblico generico", ma cluster demografici differenziati: operai metalmeccanici del Veneto, partite IVA under 35, pensionati del Sud, impiegati pubblici, studenti universitari. Ogni cluster ha profilo demografico ISTAT-compatibile, livello di engagement, canali informativi preferiti, sensibilita ai diversi temi. Reagiscono ai contenuti degli Elite e delle Istituzioni, ma anche tra loro — effetto gregge, omofilia sociale, bolle informative.

### Il Ciclo di Simulazione

Ogni simulazione si svolge in **round temporali** (tipicamente 9, ciascuno rappresentante un mese o una settimana). In ogni round:

1. **Event Injection** — Il sistema genera un evento plausibile per quel punto temporale. Non lo inventa dal nulla: si basa su precedenti storici, sul contesto attuale ricercato automaticamente via web, e su vincoli di plausibilita. Se stai simulando una riforma del lavoro, al round 3 non ti salta fuori un terremoto — ti esce uno sciopero generale o un dato ISTAT negativo o una dichiarazione di Bruxelles, perche quelli sono gli eventi coerenti con quello scenario.

2. **Reazione degli Elite** — Ogni agente Tier 1 analizza l'evento, le dichiarazioni degli altri agenti, lo stato dell'opinione pubblica, e produce una reazione. Non una reazione generica: una reazione nel *suo* stile, coerente con la *sua* storia, vincolata dalla *sua* rigidita ideologica. Un leader CGIL e un presidente di Confindustria non reagiscono allo stesso modo — e il sistema lo sa.

3. **Risposta Istituzionale** — Le istituzioni si posizionano, con i loro tempi e le loro logiche.

4. **Propagazione ai Cluster** — L'opinione pubblica assorbe, filtra, amplifica. Ogni cluster e esposto a canali diversi (chi guarda il TG1, chi sta su Twitter, chi legge Il Sole 24 Ore) e reagisce in modo diverso. Qui entrano in gioco i modelli matematici di dinamica dell'opinione.

5. **Aggiornamento dello Stato** — Posizioni, polarizzazione, coalizioni, engagement: tutto viene ricalcolato. Il round successivo parte da questo nuovo stato.

### Il Motore Matematico: Opinion Dynamics

Questo e il pezzo piu tecnico e piu importante. Le posizioni degli agenti non si muovono "a caso" o "perche l'AI ha deciso cosi". Si muovono secondo un **modello matematico di dinamica delle opinioni** con sette parametri calibrabili:

- **Anchor weight** — Quanto un agente tende a tornare alla sua posizione originale (inerzia ideologica)
- **Social weight** — Quanto e influenzato dalla media degli agenti vicini nella rete sociale
- **Event weight** — Quanto un evento esterno sposta la posizione
- **Herd weight e threshold** — Effetto gregge: quando una maggioranza supera una soglia, trascina anche i resistenti
- **Direct shift weight** — Quanto i contenuti virali degli Elite spostano direttamente i cluster piu vulnerabili
- **Anchor drift rate** — Quanto l'ancora stessa si muove nel tempo (le persone *cambiano* la propria identita, lentamente)

Questi sette parametri non sono scelti a intuito. Sono stati **calibrati su 52 scenari storici reali** — dal referendum costituzionale italiano del 2016 alla Brexit, da crisi aziendali a riforme pensionistiche europee — attraverso una grid search su 972 combinazioni di parametri. Per ogni scenario storico conosciamo l'esito reale: confrontiamo la traiettoria simulata con quella vera e minimizziamo l'errore.

Il risultato: **errore medio sull'esito finale inferiore all'11%**. Su 8 dei 52 scenari, l'errore e sotto l'1%.

Questo non e un modello teorico. E un modello calibrato empiricamente, come si fa in fisica o in epidemiologia.

### Anti-Allucinazione: Vincoli e Validazione

Il rischio di qualsiasi sistema basato su AI generativa e l'allucinazione — inventare cose. DigitalTwinSim ha un intero layer dedicato a impedirlo:

- **Position delta cap**: Nessun agente puo spostare la propria posizione di piu di 0.15 punti per round (su una scala da -1 a +1). Un politico non cambia idea da un giorno all'altro, e la simulazione rispetta questo vincolo.
- **Shock magnitude clamping**: Gli eventi non possono avere un impatto superiore a 0.6. Nessun singolo evento ribalta tutto — come nella realta.
- **Alliance validation**: Il sistema verifica che le coalizioni generate siano tra agenti che effettivamente esistono nella simulazione e che abbiano posizioni compatibili.
- **Plausibility check**: Ogni evento generato viene sottoposto a un controllo di plausibilita automatico. Se il punteggio e sotto la soglia, viene rigenerato.
- **Sentiment normalization**: La distribuzione del sentimento pubblico e vincolata a sommare sempre 100% — niente numeri magici.

### Grounding: Dati Reali, Non Invenzioni

Quando lanci una simulazione, il sistema non lavora nel vuoto:

1. **Web Research automatica** — Cerca automaticamente contesto attuale sullo scenario: articoli recenti, posizioni dichiarate, dati economici.
2. **Seed Data verificati** — Per gli scenari politici italiani, il sistema ha accesso a dati demografici ISTAT, posizioni parlamentari, risultati elettorali, dati previdenziali.
3. **RAG su documenti** — Puoi caricare documenti (PDF, DOCX, dati JSON) che diventano parte integrante del contesto. Una bozza di legge, un rapporto CNEL, un'analisi OCSE: il sistema li digerisce e li usa per ancorare la simulazione.

Gli agenti non sono "inventati dall'AI". Sono **costruiti su dati reali e poi arricchiti dall'AI** dove i dati non arrivano. La differenza e fondamentale.

---

## A Cosa Ti Serve

Te lo dico diretto: le aree della tua Commissione sono quelle dove questo strumento da il massimo, perche sono temi dove milioni di persone con interessi opposti reagiscono a catena.

### Riforma del Mercato del Lavoro

*"Il Governo propone il contratto unico a tutele progressive, eliminando i contratti a termine sotto i 12 mesi."*

La simulazione ti mostra:
- Come reagiscono le PMI del Nord-Est vs. le grandi imprese — non a parole, con traiettorie di posizionamento round per round
- Dove si spaccano CGIL, CISL e UIL (perche si spaccano — il modello lo sa)
- Chi guida il framing mediatico e con quale narrativa
- Come si sposta l'opinione dei giovani precari vs. i lavoratori gia tutelati
- Il punto esatto di massima polarizzazione — e cosa lo innesca
- Le coalizioni inattese che si formano (Confindustria + sindacati su un punto specifico? Succede piu spesso di quanto si pensi)

### AI e Occupazione

*"Un grande gruppo industriale italiano annuncia la sostituzione del 30% dei ruoli amministrativi con AI."*

Questo e il tema dei prossimi anni e la tua Commissione sara in prima linea. La simulazione permette di:
- Testare in anticipo tre risposte legislative diverse (incentivi alla riqualificazione vs. tassazione dell'automazione vs. riduzione orario) e vedere quale genera piu consenso
- Capire chi guida la narrazione e come cambia nel tempo
- Identificare i cluster demografici piu vulnerabili alla radicalizzazione

### Riforma Previdenziale

*"Superamento della Legge Fornero: Quota 42 con penalizzazioni sull'assegno."*

Puoi confrontare tre varianti — Quota 42, Quota 41, flessibilita in uscita — e vedere quale genera meno conflittualita, dove si formano le resistenze, come reagisce la Ragioneria dello Stato, come si muove l'opinione dei diversi blocchi generazionali.

### Salario Minimo

*"Salario minimo legale a 9 euro con deroga per settori coperti da CCNL."*

Un tema che hai gia gestito in Commissione. La simulazione puo mostrarti come la clausola di deroga cambia radicalmente il gioco — e confrontare scenari (con deroghe, senza deroghe, solo rafforzamento CCNL) con dati, non con opinioni.

---

## Cosa Esce dalla Simulazione

### Report Esecutivo
Documento narrativo completo, esportabile in PDF. Non tabelle e numeri: una storia. Chi si muove per primo, dove nasce il conflitto, quali coalizioni emergono, dove sono i rischi reputazionali. Pensato per essere condiviso con colleghi e staff senza bisogno di spiegazioni tecniche.

### Dashboard Interattiva
Accesso web riservato. Replay temporale round per round, analisi del singolo agente, rete di influenze dinamica, feed dei post e delle dichiarazioni con metriche di impatto. Per chi vuole andare in profondita.

### Confronto What-If
Lanci tre varianti dello stesso scenario e le confronti fianco a fianco. Quale formulazione genera meno resistenza? Quale produce piu consenso trasversale? Dove si biforcano le traiettorie? I dati grezzi sono esportabili (SQL, JSON) per analisi ulteriori.

---

## Come Si Usa

1. **Descrivi lo scenario** — in italiano, a parole tue. Puoi allegare documenti (bozze legislative, rapporti, dati).
2. **Il sistema costruisce** — in pochi minuti: ricerca contesto, genera agenti, configura la simulazione.
3. **La simulazione gira** — 5-12 round, ciascuno con eventi, reazioni, spostamenti di opinione.
4. **Leggi i risultati** — report, dashboard, confronti. Lancia varianti.

Costo per simulazione: pochi euro. Tempo: minuti.

---

## Perche Mi Fido di Questo Strumento

Te lo dico perche ci conosciamo e sai come lavoro.

Non e un prototipo da hackathon. E un sistema ingegnerizzato con:
- **14 domini tematici** calibrati indipendentemente (politico, lavoro, previdenziale, corporate, finanziario, sanitario, marketing, sport, tecnologia, energia, legale, media, educazione, ambiente)
- **52 scenari storici** di calibrazione con ground truth verificato
- **972 combinazioni di parametri** testate per ogni dominio
- **Pipeline RAG** per ingestione di documentazione proprietaria
- **Motore di opinion dynamics a 7 parametri** calibrato empiricamente
- **Layer di validazione anti-allucinazione** a 5 livelli

Viene dalla stessa logica con cui ho lavorato sulla modellistica epidemiologica — stesso rigore, stessa calibrazione empirica, stesso approccio di validazione contro dati reali. Solo che invece di prevedere curve di contagio, prevede curve di opinione.

---

## Proposta

Ti propongo una cosa semplice: **una demo dal vivo su uno scenario a tua scelta.** Un tema che stai trattando in Commissione adesso. Lo simuliamo insieme, vedi i risultati, valuti tu.

Un'ora di tempo. Se e utile ne parliamo, se non lo e ci siamo visti per un caffe.

Fammi sapere quando ti torna comodo.

Alberto

---

*Alberto Giovanni Gerli*
*Tourbillon Tech srl*
*Fellow — Universita degli Studi di Milano*
*Pubblicazioni: The Lancet, riviste internazionali peer-reviewed*

---

*Documento riservato.*
