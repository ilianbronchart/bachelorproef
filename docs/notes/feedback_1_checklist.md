
### Algemeen
- [ ] Verwijder **alle** `\newline` en `\par` uit je bronbestand.  
- [ ] Begin een nieuwe paragraaf door één lege regel tussen de tekst te laten.

### Samenvatting
- [x] Verwijder de huidige samenvatting: schrijf deze pas **aan het einde** (nadat conclusies vastliggen).  
- [ ] Zorg dat de samenvatting de conclusies omvat en geen inleiding is.  
- [ ] Schrijf de samenvatting volledig in de **verleden tijd**.

### 1.2 Onderzoeksvraag
- [ ] Corrigeer de woordafbreking van **“Zorglab‑specifieke”** volgens de HOGENT‑hyphenatiegids.  
- [ ] Controleer in de gehele tekst of samengestelde woorden mét koppeltekens correct afgebroken worden.  
- [ ] Laat je deelvragen niet alleen in het oplossingsdomein, maar werk ze ook uit in het probleemdomein.

### 2.1.1 Soorten eyetrackers
- [ ] Vervang de placeholder **“Sectie ??”** door het correcte label (bv. `\ref{sec:…}`).

### 2.1.3 Bestaande oplossingen voor eyetracking data‑analyse
- [ ] Voeg de volgende Tobii‑artikels toe als `@online` bronnen in je `.bib`:
  - Understanding Tobii Pro Lab eye‑tracking metrics  
  - Visualizations for Tobii Pro Lab  

### 2.2 Computer Vision
- [ ] Pas bij **Figuur 2.1** de bronvermelding aan: geen URL, maar een APA‑verwijzing met `\autocite{}`.  
- [ ] Breid het bijschrift van de figuur uit met voldoende context, zodat de lezer niet in de hoofdtekst hoeft te zoeken.

### 3.1 Tijdplanning
- [ ] Schrap of minimaliseer de uitgebreide tijdplanning (het onderzoek is afgerond bij indiening).  
- [ ] Behoud kort de fasenbeschrijving indien ze tot het einde zijn aangehouden.  
- [ ] Werk de methodologie later verder uit (pas aan naarmate je onderzoek vordert).  
- [ ] Schrijf alles in de **verleden tijd**.

### Bibliografie
- [ ] **Cederin & Bremberg (2023)**: voeg ontbrekende instelling/institutie toe.  
- [ ] In `bachproef.bib:10`: verwijder het voorvoegsel `https://doi.org/` uit het DOI‑veld.  
- [ ] Vervang waar mogelijk `Year` door `Date` en `Journal` door `Journaltitle` (BibLaTeX‑conventie).  
- [ ] In `bachproef.bib:63`: verwijder het lege `pages`‑veld.  
- [ ] **Rublee et al. (2011)** (`bachproef.bib:70`):
  - Verplaats de “Proceedings of …” naar het `booktitle`‑veld.  
  - Gebruik een en‑dash (`--`) tussen paginanummers.  
- [ ] Controleer bronclassificaties:
  - **Lindeberg (2012)**: moet `@online` zijn (niet `@inbook`); voeg `url` + `urldate` toe; probeer indien mogelijk een primaire bron.  
  - **Kulyk (2023)**: wijzig naar `@online`; voeg `urldate` toe.  
  - Herclassificeer andere `@misc`‑items naar specifieker type of `@online`; voeg overal `urldate` toe.  
- [ ] Draai tot slot BibLa (via JabRef‑plugin of command‑line) voor extra validaties en suggesties (zie HOGENT‑gids).
