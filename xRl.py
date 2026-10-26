import random
from collections import deque
import numpy as np #macierze
import torch #sieci neuronowe
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import time

# generowanie wykresu
def plot_rewards(rewards, output_path):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(rewards) + 1), rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total reward")
    plt.title("Training rewards")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


# NEURAL NETWORK

#definicja sieci neuronowej Q (core)
class TrafficDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(TrafficDQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

#przetwarzanie danych wejściowych przez sieć
    def forward(self, x):
        return self.fc(x)

#bufor pamięci doświadczeń
class ReplayBuffer:
    def __init__(self, capacity): #maksymalna pojemność bufora
        self.buffer = deque(maxlen=capacity)

#wrzuc dane do bufora
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

#losowe próbkowanie z bufora
    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return (
            np.array(state, dtype=np.float32),
            np.array(action, dtype=np.int64),
            np.array(reward, dtype=np.float32),
            np.array(next_state, dtype=np.float32),
            np.array(done, dtype=np.uint8)
        )

    def __len__(self): #pomocnicza funkcja do sprawdzania długości bufora
        return len(self.buffer)

import traci #łączenie z sumo

#SUMO

class SumoTrafficEnv: #podłączenie plików konfiguracyjnych sumo do naszego kodu
    def __init__(self, sumocfg_file, tls_id, incoming_lanes, use_gui=False):
        self.sumocfg_file = sumocfg_file
        self.tls_id = tls_id
        self.incoming_lanes = incoming_lanes
        self.use_gui = use_gui
        self.sumo_binary = "sumo-gui" if use_gui else "sumo"

    def start(self):
        traci.start([self.sumo_binary, "-c", self.sumocfg_file, "--start", "--no-step-log", "true"])

    def close(self):
        try:
            traci.close()
        except traci.TraCIException:
            pass

    def get_state(self):
        # pobieramy liczbę zatrzymanych pojazdów lub czas oczekiwania na pasach wwozowych
        state = [traci.lane.getWaitingTime(lane) for lane in self.incoming_lanes]
        return np.array(state, dtype=np.float32)

    def apply_action(self, action):
        # zmiana fazy sygnalizacji świetlnej
        traci.trafficlight.setPhase(self.tls_id, int(action))

    def step(self, action, delta_time=10): #zmiana fazy
        self.apply_action(action)
        
        reward = 0
        for _ in range(delta_time): #dzialanie fazy przez chwile i obliczenie nagrody
            traci.simulationStep()
            # nagroda jako ujemna suma czasu oczekiwania na pasach wwozowych w tym kroku
            current_waiting = sum(traci.lane.getWaitingTime(lane) for lane in self.incoming_lanes)
            reward -= current_waiting

        next_state = self.get_state()
        
        # sprawdzamy czy symulacja się skończyła
        done = traci.simulation.getMinExpectedNumber() <= 0
        return next_state, reward, done

#TRAINING FUNCTION

def train_dqn():
    # konfiguracja, podłączenie plików sumo i ustawienia środowiska 
    sumocfg_file = "config.sumocfg"  # scieżka .sumocfg
    tls_id = "J2"     # ID sygnalizacji świetlnej w pliku .net.xml
    incoming_lanes = ["E72_0", "E72_1", "E12_0", "E12_1", "E52_0", "E52_1", "E32_0", "E32_1"] # pasy wwozowe do skrzyżowania (plik .net.xml)
    
    num_phases = 4                         # Liczba dostępnych faz świateł
    state_dim = len(incoming_lanes)
    action_dim = num_phases
    
    # Hiperparametry RL
    episodes = 100 #liczba epizodów do wytrenowania modelu
    gamma = 0.99 #współczynnik 'patrzenia w przyszłość' dla nagród
    batch_size = 32 #rozmiar próbki do uczenia sieci neuronowej (batch size)
    lr = 0.001 #współczynnik uczenia się (learning rate) dla optymalizatora Adam
    epsilon = 1.0 #współczynnik losowości (eksploracja) im większy tym bardziej losowe akcje, im mniejszy tym bardziej deterministyczne (wybór najlepszej akcji)
    epsilon_decay = 0.95 #współczynnik zmniejszania losowości (eksploracji) po każdym epizodzie
    min_epsilon = 0.05 #minimalny współczynnik losowości (eksploracji) po którym nie zmniejszamy już losowości (eksploracji)
    
    env = SumoTrafficEnv(sumocfg_file, tls_id, incoming_lanes, use_gui=False)
    
    policy_net = TrafficDQN(state_dim, action_dim)
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    criterion = nn.MSELoss()
    memory = ReplayBuffer(capacity=10000)
    episode_rewards = []
    
    for episode in range(episodes):
        env.start()
        state = env.get_state()
        total_reward = 0
        done = False
        
        while not done:
            # wybór akcji (epsilon-greedy)
            if random.random() < epsilon:
                action = random.randint(0, action_dim - 1)
            else:
                with torch.no_grad(): #if random.random() >= epsilon: weź najlepszą akcję z sieci neuronowej
                    q_values = policy_net(torch.FloatTensor(state).unsqueeze(0))
                    action = torch.argmax(q_values).item()
            
            next_state, reward, done = env.step(action, delta_time=10) #zmiana świateł i czekanie 7 sekund
            memory.push(state, action, reward, next_state, done) #zapisanie danych
            
            state = next_state
            total_reward += reward
            
            # uczenie sieci, gdy w buforze jest wystarczająco danych
            if len(memory) >= batch_size:
                states, actions, rewards, next_states, dones = memory.sample(batch_size)

                #zapisanie w tablicach pytorch
                state_tensor = torch.FloatTensor(states)
                next_state_tensor = torch.FloatTensor(next_states)
                action_tensor = torch.LongTensor(actions)
                reward_tensor = torch.FloatTensor(rewards)
                done_tensor = torch.FloatTensor(dones)
                
                # oblicznie q
                q_values = policy_net(state_tensor)
                state_action_values = q_values.gather(1, action_tensor.unsqueeze(1)).squeeze(1)
                
                # równanie Bellmana
                with torch.no_grad():
                    next_q_values = policy_net(next_state_tensor)
                    max_next_q_values = torch.max(next_q_values, dim=1)[0]
                    target_q_values = reward_tensor + (gamma * max_next_q_values * (1 - done_tensor))

                loss = criterion(state_action_values, target_q_values)

                #korygowanie podejmowanych decyzji w sieci neuronowej
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        env.close()
        episode_rewards.append(total_reward)
        
        # aktualizacja epsilon
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        print(f"Epizod {episode + 1}/{episodes} | Całkowita nagroda: {total_reward:.2f} | Epsilon: {epsilon:.2f}")
        
    # zapis modelu
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    rewards_graph_name = f"training_rewards_{timestamp}_with_{episodes}_episodes.png"
    plot_rewards(episode_rewards, rewards_graph_name)
    print(f"Wykres nagród zapisano jako {rewards_graph_name}")

    model_name = f'traffic_dqn_model_{timestamp}_with_{episodes}_episodes.pth'
    torch.save(policy_net.state_dict(), model_name)
    print("Model został wytrenowany i zapisany pomyślnie jako {model_name}")

if __name__ == "__main__":
    train_dqn()