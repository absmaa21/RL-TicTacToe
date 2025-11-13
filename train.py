from game import TicTacToe
from agent import QLearningAgent
import random

EPSILON_DECAY = 0.99995
MIN_EPSILON = 0.05
EPISODES = 50000

def make_state_key(board_tuple, current_player):
    # represent state as (board_tuple, current_player) so Q differentiates turns
    return (board_tuple, current_player)

def train(episodes=EPISODES, save_path="qtable.pkl"):
    env = TicTacToe()
    agent = QLearningAgent(epsilon=1.0)

    for ep in range(episodes):
        board = env.reset()
        done = False
        while not done:
            state_key = make_state_key(board, env.current_player)
            legal = env.legal_actions()
            action = agent.get_action(state_key, legal, training=True)

            # agent plays
            after_agent_board, reward_agent_move, done, info = env.step(action)

            if done:
                # terminal immediately after agent move
                next_state_key = make_state_key(after_agent_board, env.current_player)
                agent.update(state_key, action, reward_agent_move, next_state_key, [], True)
                board = after_agent_board
                break

            # opponent (random) plays
            opp_action = random.choice(env.legal_actions())
            after_opp_board, reward_opp_move, done, info = env.step(opp_action)

            # compute agent-centric reward AFTER opponent move
            if done:
                # game ended because of opponent move or draw
                if env.winner == 1:
                    final_reward = 1
                elif env.winner == -1:
                    final_reward = -1
                else:
                    final_reward = 0
            else:
                final_reward = 0

            next_state_key = make_state_key(after_opp_board, env.current_player)
            agent.update(state_key, action, final_reward, next_state_key, env.legal_actions(), done)
            board = after_opp_board

        # decay epsilon slowly
        agent.epsilon = max(MIN_EPSILON, agent.epsilon * EPSILON_DECAY)
        if ep % 5000 == 0:
            print(f"Episode {ep}/{episodes}, epsilon={agent.epsilon:.3f}")

    agent.save(save_path)
    print(f"Training completed and saved to {save_path}")
    return agent

if __name__ == "__main__":
    train()
