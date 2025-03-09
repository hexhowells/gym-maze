# gym-maze
Gymnasium for randomly generated 3D maze games

![Maze](https://github.com/hexhowells/gym-maze/blob/main/gameplay.gif)

## Example code
```python
from gym_maze import MazeEnv


env = MazeEnv(early_stop_threshold=2_000)
obs = env.reset()
done = False

while not done:
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    env.render()

env.close()
```
