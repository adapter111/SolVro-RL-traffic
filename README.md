# SolVro-RL-traffic

Wymagania:
Python (zalecana wersja: 3.13.17)
SUMO (Simulation of Urban MObility)
Zmienna środowiskowa SUMO_HOME

INSTRUKCJA

1. Sklonuj repo
   ```bash
   git clone https://github.com/adapter111/SolVro-RL-traffic
   cd SolVro-RL-traffic
   ```

3. Skonfiguruj zmienną środowiskową SUMO_HOME
   2.1 Sprawdź czy zmienna SUMO_HOME jest już ustawiona
     ```pwsh
       echo $env:SUMO_HOME
     ```
   2.2 Jeżeli nie jest, sprawdź, gdzie zainstalowane jest twoje SUMO
     ```cmd
     where sumo
     ```
   2.3 Ustaw zmienną SUMO_HOME
     ```pwsh
     $env:SUMO_HOME = "C:\...\..."
     ```

4. Zainstaluj wymagane biblioteki
   ```bash
   pip install -r requirements.txt
   ```

* `xRL.py` - Plik zawierający kod używany do treningu i zapisania wyników.
* `xVisualize.py` - Plik, który umożliwia odpalenie wizualizacji, na podstawie jednego z czterech wytrenowanych modeli (domyślnie ustawiono model wytrenowany przez 10 epizodów).
Aby zmienić model wystarczy zamienić nazwę pliku modelu w 17 linijce kodu, przy policy_net.load_state_dict
* `config.sumocfg` - Główny plik konfiguracyjny symulatora SUMO.
* `network.net.xml` - Definicja infrastruktury drogowej (skrzyżowania, pasy).
* `config.rou.xml` - Definicja tras i natężenia ruchu pojazdów.
* `traffic_dqn_model_...` - Zapisane wagi wytrenowanych modeli deweloperskich.

Model trenowany był jedynie na skrzyżowaniu centralnym (ID=J2), z 8 pasami wwozowymi.

Nagrodą dla modelu jest ujemna suma czasu oczekiwania na zakorkowanych pasach.

WYNIKI TRENINGU

traffic_dqn_model_... - to cztery pliki modeli.
Poniżej wykresy wyników każdego z nich.

1. _with_10_episodes - Lekko chaotyczny wykres, natomiast widać trend wzrostowy. Najwyższa nagroda około -60 000
   <img width="1000" height="500" alt="training_rewards_20261026-185906_with_10_episodes" src="https://github.com/user-attachments/assets/fc9767ea-e085-4663-9e5e-679e07dd288d" />

2. _with_25_episodes - Wyraźnie widać progres modelu. Najwyższa nagroda około -21 000
   <img width="1000" height="500" alt="training_rewards_20261026-190753_with_25_episodes" src="https://github.com/user-attachments/assets/d086e70b-ffed-4ee3-aa1c-db0017515cf3" />

3. _with_50_episodes - Widać szybki wzrost, potem spowolnienie rozwoju. Najwyższa nagroda około -18 000
   <img width="1000" height="500" alt="training_rewards_20261026-191459_with_50_episodes" src="https://github.com/user-attachments/assets/7179d515-a6d1-4a25-b378-44d3ce0537c9" />

4. _with_100_episodes - Podobnie jak wyżej. Najwyższa nagroda około -12 400
   <img width="1000" height="500" alt="training_rewards_20261026-192605_with_100_episodes" src="https://github.com/user-attachments/assets/1dd082d5-0f77-4e67-82eb-105fdfd0a18d" />
