# UML Diagrams

This document provides Mermaid UML diagrams to explain how the system works at a high level.

## Call Flow (Demo Loop)

```mermaid
flowchart TD
  "train/demo.py::main()" --> "pygame.event.get()"
  "train/demo.py::main()" --> "QNetwork.forward()"
  "QNetwork.forward()" --> "train/independent_dqn_pytorch.py::QNetwork"

  "train/demo.py::main()" --> "env/swarm_env.py::SwarmEnv.step()"
  "env/swarm_env.py::SwarmEnv.step()" --> "env/swarm_env.py::DynamicsDriver.apply()"
  "env/swarm_env.py::DynamicsDriver.apply()" --> "env/swarm_env.py::TankKinematicsDriver.apply()"
  "env/swarm_env.py::DynamicsDriver.apply()" --> "env/swarm_env.py::HovercraftDriver.apply()"
  "env/swarm_env.py::SwarmEnv.step()" --> "env/swarm_env.py::_handle_collisions()"
  "env/swarm_env.py::SwarmEnv.step()" --> "env/swarm_env.py::_handle_targets()"
  "env/swarm_env.py::SwarmEnv.step()" --> "env/swarm_env.py::_update_pheromone()"
  "env/swarm_env.py::SwarmEnv.step()" --> "env/swarm_env.py::_get_obs()"

  "train/demo.py::main()" --> "env/swarm_env.py::SwarmEnv.render()"
  "env/swarm_env.py::SwarmEnv.render()" --> "env/swarm_env.py::_ensure_pygame()"
  "env/swarm_env.py::SwarmEnv.render()" --> "pygame.display.flip()"
  "env/swarm_env.py::SwarmEnv.render()" --> "pygame.time.Clock.tick()"
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
