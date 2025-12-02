from physics_sim import simulate_rocket
from graph import create_graphs

df = simulate_rocket(T_total=20, save=True, save_path="datasets/rocket_flight_data.csv")
create_graphs(data=df)