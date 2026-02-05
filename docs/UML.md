# UML Diagrams

This document provides Mermaid UML diagrams to explain how the system works at a high level.

## Call Flow (Demo Loop)

```mermaid
flowchart TD
  Demo["train/demo.py::main()"] --> PygameEvents["pygame.event.get()"]
  Demo --> QForward["QNetwork.forward()"]
  QForward --> QClass["train/independent_dqn_pytorch.py::QNetwork"]

  Demo --> Step["env/swarm_env.py::SwarmEnv.step()"]
  Step --> Apply["env/swarm_env.py::DynamicsDriver.apply()"]
  Apply --> Tank["env/swarm_env.py::TankKinematicsDriver.apply()"]
  Apply --> Hover["env/swarm_env.py::HovercraftDriver.apply()"]
  Step --> Collide["env/swarm_env.py::_handle_collisions()"]
  Step --> Targets["env/swarm_env.py::_handle_targets()"]
  Step --> Phero["env/swarm_env.py::_update_pheromone()"]
  Step --> Obs["env/swarm_env.py::_get_obs()"]

  Demo --> Render["env/swarm_env.py::SwarmEnv.render()"]
  Render --> Ensure["env/swarm_env.py::_ensure_pygame()"]
  Render --> Flip["pygame.display.flip()"]
  Render --> Tick["pygame.time.Clock.tick()"]
```

## Sequence Diagram (One Frame)

```mermaid
sequenceDiagram
  participant Demo as train/demo.py::main()
  participant Policy as QNetwork.forward()
  participant Env as env/swarm_env.py::SwarmEnv
  participant Driver as DynamicsDriver.apply()
  participant Render as SwarmEnv.render()
  participant PG as PyGame

  Demo->>PG: pygame.event.get()
  Demo->>Policy: obs[i]
  Policy-->>Demo: action
  Demo->>Env: step(actions)
  Env->>Driver: apply(state, action)
  Driver-->>Env: proposed_state
  Env-->>Demo: obs, rewards, done
  Demo->>Render: render()
  Render->>PG: draw + flip + tick
```

## Class Diagram (Core Runtime)

```mermaid
classDiagram
  class SwarmEnv {
    +reset(seed)
    +step(actions)
    +render(mode, fps)
    +close()
    -_ensure_pygame()
    -_handle_collisions()
    -_handle_targets()
    -_update_pheromone()
    -_get_obs()
  }

  class SwarmConfig
  class AgentState

  class DynamicsDriver {
    +apply(state, action, dt, cfg, rng)
  }
  class TankKinematicsDriver
  class HovercraftDriver

  class QNetwork {
    +forward(obs)
  }

  SwarmEnv --> SwarmConfig : config
  SwarmEnv o-- AgentState : agents
  SwarmEnv --> DynamicsDriver : driver
  DynamicsDriver <|-- TankKinematicsDriver
  DynamicsDriver <|-- HovercraftDriver
  QNetwork ..> SwarmEnv : acts on obs
```

## State Update (RL Step)

```mermaid
flowchart LR
  A["Actions (per agent)"] --> B["DynamicsDriver.apply()"]
  B --> C["Collision Check"]
  C --> D["Target Collection"]
  D --> E["Pheromone Update"]
  E --> F["Observations"]
  F --> G["Rewards + Done flags"]
```

## Training Flow (Independent DQN)

```mermaid
flowchart TD
  Start["train/independent_dqn_pytorch.py::main()"] --> InitEnv["Initialize SwarmEnv (headless)"]
  InitEnv --> InitNets["Init QNetwork(s) + target network(s)"]
  InitNets --> InitBuf["Init replay buffer(s)"]
  InitBuf --> Loop["Training loop (total steps)"]

  Loop --> Act["Epsilon-greedy action selection"]
  Act --> Step["env.step(actions)"]
  Step --> Store["Store transition in replay buffer"]
  Store --> TrainCheck["If enough data: sample batch"]
  TrainCheck --> Update["Compute TD loss + optimizer step"]
  Update --> TargetUpdate["Periodic target network update"]
  TargetUpdate --> Save["Optional checkpoint save"]
  Save --> DoneCheck["If episode done: env.reset()"]
  DoneCheck --> Loop
```

## Training Sequence (Single Step)

```mermaid
sequenceDiagram
  participant Trainer as train/independent_dqn_pytorch.py::main()
  participant Env as env/swarm_env.py::SwarmEnv
  participant Policy as QNetwork.forward()
  participant Buffer as ReplayBuffer

  Trainer->>Policy: obs -> Q-values
  Policy-->>Trainer: Q-values
  Trainer->>Env: step(actions)
  Env-->>Trainer: next_obs, rewards, done
  Trainer->>Buffer: add(transition)
  Trainer->>Buffer: sample(batch)
  Trainer->>Policy: compute loss + backprop
```
