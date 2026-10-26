import torch
from xRl import TrafficDQN
from xRl import SumoTrafficEnv

#VISUALIZATION FUNCTION

def visualize_model():
    sumocfg_file = "config.sumocfg"
    tls_id = "J2"
    incoming_lanes = ["E72_0", "E72_1", "E12_0", "E12_1", "E52_0", "E52_1", "E32_0", "E32_1"]
    
    state_dim = len(incoming_lanes)
    action_dim = 4
    
    # ładowanie wytrenowanego modelo
    policy_net = TrafficDQN(state_dim, action_dim)
    policy_net.load_state_dict(torch.load("traffic_dqn_model_20261026-185906_with_10_episodes.pth"))
    policy_net.eval()
    
    # Uruchomienie z interfejsem graficznym SUMO GUI
    env = SumoTrafficEnv(sumocfg_file, tls_id, incoming_lanes, use_gui=True)
    env.start()
    
    state = env.get_state()
    done = False
    
    print("Uruchamianie wizualizacji wytrenowanego modelu w SUMO GUI...")
    while not done:
        with torch.no_grad():
            q_values = policy_net(torch.FloatTensor(state).unsqueeze(0))
            action = torch.argmax(q_values).item()
            
        state, reward, done = env.step(action, delta_time=10)
        
    env.close()

if __name__ == "__main__":
    visualize_model()