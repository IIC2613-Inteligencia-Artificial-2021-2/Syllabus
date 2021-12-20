import random
import numpy as np
from game import SnakeGameAI, Direction, Point
from helper import plot
import pickle
from load_qtables import export_qtable, import_qtable

# Hiperparámetros
LR = 0.05
NUM_EPISODES = 10_000_000
DISCOUNT_RATE = 0
MAX_EXPLORATION_RATE = 1
MIN_EXPLORATION_RATE = 0.0001
EXPLORATION_DECAY_RATE = 0.005

class Agent:
    # Esta clase posee al agente y define sus comportamientos.

    def __init__(self):
        # Creamos la q_table y la inicializamos en 0.

        # Estas son todas las posibles posiciones de la comida.
        food_pos = [[1,0,0,0],[1,0,1,0],[1,0,0,1],
                    [0,1,0,0],[0,1,1,0],[0,1,0,1],
                    [0,0,0,1],[0,0,1,0]]
        # Estas son todas las posibles direcciones del agente.
        dir_pos = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
        self.states_index = dict() # Diccionario con el estado del agente como key
                                   # y su indice en la q_table como value.
        index = 0
        for d_s in range(2):
            for d_r in range(2):
                for d_l in range(2):
                    for dir in dir_pos: 
                        for food in food_pos:
                            self.states_index[tuple([d_s, d_l, d_r] + dir + food)] = index
                            index += 1 
        # Inicializamos la q_table en 0.
        self.q_table = np.zeros((2*2*2*4*8, 3))
        # Inicializamos los juegos realizados por el agente en 0.
        self.n_games = 0
        # Inicializamos el exploration rate.
        self.EXPLORATION_RATE = MIN_EXPLORATION_RATE + (MAX_EXPLORATION_RATE - MIN_EXPLORATION_RATE) * np.exp(-EXPLORATION_DECAY_RATE*self.n_games)

    def get_state(self, game):
        # Este método consulta al juego por el estado del agente
        # y lo retorna como una tupla.

        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
            ]

        return tuple(state)

    def get_action(self, state):
        # Este método recibe una estado del agente y retorna una
        # tupla con el indice de la acción correspondiente y una lista que la
        # representa.

        move = 0
        if random.uniform(0, 1) < self.EXPLORATION_RATE: # Exploramos
            move = random.randint(0, 2)
        else: # Explotamos
            # Si los valores para este estado siguen en 0, tomamos una acción random para no sesgar al agente.
            if not np.any(self.q_table[self.states_index[state],:]): 
                move = random.randint(0, 2)
            else: 
                move = np.argmax(self.q_table[self.states_index[state],:])
        return move

def train():
    # Esta función es la encargada de entrenar al agente.

    # Las siguientes variables nos permitirán llevar registro del
    # desempeño del agente.
    plot_scores = []
    plot_mean_scores = []
    mean_score = 0
    total_score = 0
    record = 0
    period_steps = 0
    period_score = 0

    # Instanciamos al agente o lo cargamos desde un pickle.
    # agent = Agent()

    agent = pickle.load(open("model/agent_6.p", "rb"))
    # Instanciamos el juego. El bool 'vis' define si queremos visualizar el juego o no.
    # Visualizarlo lo hace mucho más lento.
    game = SnakeGameAI(vis=True)
    # Inicializamos los pasos del agente en 0.
    steps = 0

    while True:
        # Obtenemos el estado actual.
        state = agent.get_state(game)
        # Generamos la acción correspondiente al estado actual.
        move = agent.get_action(state)
        # Ejecutamos la acción.
        reward, done, score = game.play_step(move)
        # Obtenemos el nuevo estado.
        state_new = agent.get_state(game)
        # Actualizamos la q-table.
        agent.q_table[agent.states_index[state], move] = agent.q_table[agent.states_index[state], move] * (1 - LR) + \
            LR * (reward + DISCOUNT_RATE * np.max(agent.q_table[agent.states_index[state_new],:]))
        
        if done:
            # En caso de terminar el juego.

            # Actualizamos el exploration rate.
            agent.EXPLORATION_RATE = MIN_EXPLORATION_RATE + \
                (MAX_EXPLORATION_RATE - MIN_EXPLORATION_RATE) * np.exp(-EXPLORATION_DECAY_RATE*agent.n_games)
            # Reiniciamos el juego.
            game.reset()
            # Actualizamos los juegos jugados por el agente.
            agent.n_games += 1
            # Imprimimos el desempeño del agente cada 100 juegos.
            if agent.n_games % 100 == 0:
                # pickle.dump(agent, open("model/agent_6.p", "wb"))
                print('Game', agent.n_games, 'Mean Score', period_score/100, 'Record:', record, "EXP_RATE:", agent.EXPLORATION_RATE, "STEPS:", period_steps/100)
                # pickle.dump([plot_scores, plot_mean_scores], open("model/scores6.p", "wb"))      
                record = 0
                period_score = 0
                period_steps = 0
            # Actualizamos el record del agente.
            if score > record:
                record = score
            
            # Actualizamos nuestros indicadores.
            period_steps += steps
            period_score += score
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            steps = 0
            # Descomentar la linea de abajo en caso de querer ver gráficos en vivo.
            # plot(plot_scores, plot_mean_scores)
            
            # En caso de alcanzar el máximo de juegos cerramos el loop. 
            if agent.n_games == NUM_EPISODES:
                break
        else:
            steps += 1
                


if __name__ == '__main__':
    train()