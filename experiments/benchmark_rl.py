"""Experiment 6: PPO on CartPole-v1 with the time split into environment / inference / update.

Where the time goes (per PPO iteration; all synchronised on the accelerator):
  env_s    - gymnasium `envs.step` (always CPU, identical on every backend)
  infer_s  - policy forward pass for acting, incl. host->device copy of the observation and the copy of
             logits/values back to the CPU (this round trip is why tiny policies rarely benefit from a GPU)
  gae_s    - advantage estimation (CPU)
  update_s - PPO update epochs on the device, incl. host->device copy of the rollout
Actions are sampled on the CPU from the device's logits with the seeded CPU generator, so runs on
different backends face the same random stream (their trajectories can still drift through
floating-point differences).

Throughput definitions stored per iteration (`extra`):
  env_steps_per_sec_env_only  = env steps / env_s           (simulator speed)
  env_steps_per_sec_end2end   = env steps / total iteration (the row's `throughput`)
  train_steps_per_sec         = gradient steps / update_s
"""
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from src.devices import assert_on_device  # noqa: E402
from src.metrics import MemoryTracker  # noqa: E402
from src.models import count_params  # noqa: E402
from src.runner import RunContext, check_finite, experiment_main  # noqa: E402
from src.timing import DeviceTimer  # noqa: E402

EXPERIMENT = "rl"


def mlp(inp: int, hidden: int, out: int, out_std: float) -> nn.Sequential:
    net = nn.Sequential(nn.Linear(inp, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, out))
    for m in net:
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, np.sqrt(2))
            nn.init.zeros_(m.bias)
    nn.init.orthogonal_(net[-1].weight, out_std)
    return net


class Agent(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int):
        super().__init__()
        self.actor = mlp(obs_dim, hidden, n_actions, 0.01)
        self.critic = mlp(obs_dim, hidden, 1, 1.0)

    def forward(self, obs):
        return self.actor(obs), self.critic(obs).squeeze(-1)


def make_envs(name: str, n: int, seed: int):
    envs = gym.make_vec(name, num_envs=n, vectorization_mode="sync",
                        vector_kwargs={"autoreset_mode": gym.vector.AutoresetMode.SAME_STEP})
    obs, _ = envs.reset(seed=seed)
    return envs, obs


def gae(rewards, values, dones, next_value, gamma, lam):
    T = rewards.shape[0]
    adv = torch.zeros_like(rewards)
    last = torch.zeros_like(next_value)
    for t in reversed(range(T)):
        nv = next_value if t == T - 1 else values[t + 1]
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * nv * nonterminal - values[t]
        last = delta + gamma * lam * nonterminal * last
        adv[t] = last
    return adv, adv + values


def run_config(ctx: RunContext, hidden: int) -> None:
    cfg, dev = ctx.exp_cfg, ctx.device
    N, T = cfg["num_envs"], cfg["rollout_steps"]
    torch.manual_seed(ctx.seed)
    envs, obs = make_envs(cfg["env"], N, ctx.seed)
    obs_dim, n_actions = envs.single_observation_space.shape[0], int(envs.single_action_space.n)
    agent = Agent(obs_dim, n_actions, hidden)
    n_params = count_params(agent)
    agent.to(dev)
    assert_on_device(dev, agent)
    opt = torch.optim.Adam(agent.parameters(), lr=cfg["lr"], eps=1e-5)
    common = dict(model=f"ppo_mlp{hidden}", dataset=cfg["env"], batch_size=cfg["minibatch_size"], precision="fp32", hidden=hidden,
                  n_params=n_params, num_envs=N, rollout_steps=T)

    # kernel warm-up on a discarded copy: never touches the training state
    import copy
    tmp = copy.deepcopy(agent)
    for shape in (N, cfg["minibatch_size"]):
        o = torch.randn(shape, obs_dim, device=dev)
        lg, v = tmp(o)
        (lg.sum() + v.sum()).backward()
        _ = lg.cpu()
    ctx.sync()
    del tmp

    ep_ret, running = [], np.zeros(N)
    batch = N * T
    mb = cfg["minibatch_size"]
    total_env_steps = 0
    with MemoryTracker(ctx.backend) as mem:
        for it in range(cfg["ppo_iterations"]):
            obs_buf = torch.zeros(T, N, obs_dim)
            act_buf, logp_buf = torch.zeros(T, N, dtype=torch.long), torch.zeros(T, N)
            rew_buf, done_buf, val_buf = torch.zeros(T, N), torch.zeros(T, N), torch.zeros(T, N)
            env_s = infer_s = 0.0
            next_obs = torch.as_tensor(obs, dtype=torch.float32)
            for t in range(T):
                obs_buf[t] = next_obs
                with DeviceTimer(ctx.backend) as tm:
                    with torch.no_grad():
                        logits, value = agent(next_obs.to(dev))
                        logits, value = logits.cpu(), value.cpu()
                infer_s += tm.elapsed_ms / 1000
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                act_buf[t], logp_buf[t], val_buf[t] = action, dist.log_prob(action), value
                t0 = time.perf_counter()
                obs, reward, term, trunc, _ = envs.step(action.numpy())
                env_s += time.perf_counter() - t0
                done = np.logical_or(term, trunc)
                rew_buf[t], done_buf[t] = torch.as_tensor(reward, dtype=torch.float32), torch.as_tensor(done, dtype=torch.float32)
                running += reward
                for i in np.flatnonzero(done):
                    ep_ret.append(running[i])
                    running[i] = 0.0
                next_obs = torch.as_tensor(obs, dtype=torch.float32)
            total_env_steps += batch

            t0 = time.perf_counter()
            with torch.no_grad():
                with DeviceTimer(ctx.backend) as tm:
                    _, next_value = agent(next_obs.to(dev))
                    next_value = next_value.cpu()
                infer_s += tm.elapsed_ms / 1000
            adv, ret = gae(rew_buf, val_buf, done_buf, next_value, cfg["gamma"], cfg["gae_lambda"])
            gae_s = time.perf_counter() - t0 - tm.elapsed_ms / 1000

            b_obs, b_act, b_logp = obs_buf.reshape(-1, obs_dim), act_buf.reshape(-1), logp_buf.reshape(-1)
            b_adv, b_ret = adv.reshape(-1), ret.reshape(-1)
            perms = [torch.randperm(batch) for _ in range(cfg["update_epochs"])]  # CPU generator
            pl, vl, ent, grad_steps = [], [], [], 0
            with DeviceTimer(ctx.backend) as tu:
                d_obs, d_act, d_logp, d_adv, d_ret = (x.to(dev) for x in (b_obs, b_act, b_logp, b_adv, b_ret))
                for perm in perms:
                    for s in range(0, batch, mb):
                        idx = perm[s:s + mb].to(dev)
                        logits, newv = agent(d_obs[idx])
                        dist = torch.distributions.Categorical(logits=logits)
                        ratio = (dist.log_prob(d_act[idx]) - d_logp[idx]).exp()
                        a = d_adv[idx]
                        a = (a - a.mean()) / (a.std() + 1e-8)
                        p_loss = torch.max(-a * ratio, -a * ratio.clamp(1 - cfg["clip_coef"], 1 + cfg["clip_coef"])).mean()
                        v_loss = 0.5 * ((newv - d_ret[idx]) ** 2).mean()
                        e = dist.entropy().mean()
                        loss = p_loss + cfg["vf_coef"] * v_loss - cfg["ent_coef"] * e
                        opt.zero_grad(set_to_none=True)
                        loss.backward()
                        nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                        opt.step()
                        pl.append(p_loss.detach()); vl.append(v_loss.detach()); ent.append(e.detach())
                        grad_steps += 1
                pl_f, vl_f, ent_f = (float(torch.stack(x).mean().item()) for x in (pl, vl, ent))
            update_s = tu.elapsed_ms / 1000
            check_finite(pl_f + vl_f, f"(iteration {it})")

            total_s = env_s + infer_s + gae_s + update_s
            recent = ep_ret[-20:]
            ctx.record(**common, phase="measure", iteration=it, latency_ms=total_s * 1000, throughput=batch / total_s,
                       throughput_unit="env_steps/s (end-to-end)", loss=pl_f + vl_f, policy_loss=pl_f, value_loss=vl_f, entropy=ent_f,
                       env_s=env_s, infer_s=infer_s, gae_s=gae_s, update_s=update_s, env_steps=batch, grad_steps=grad_steps,
                       env_steps_per_sec_env_only=batch / env_s, env_steps_per_sec_end2end=batch / total_s,
                       train_steps_per_sec=grad_steps / update_s, mean_episode_return=(statistics.fmean(recent) if recent else None),
                       episodes_finished=len(ep_ret))
    assert_on_device(dev, agent)
    envs.close()
    ctx.record(**common, phase="config", memory_mb=mem.result["memory_mb"], memory_kind=mem.result["memory_kind"],
               total_env_steps=total_env_steps, **{k: v for k, v in mem.result.items() if k not in ("memory_mb", "memory_kind")})
    print(f"    hidden={hidden:4d}: last-iter env {env_s * 1000:6.0f} ms  infer {infer_s * 1000:6.0f} ms  update {update_s * 1000:6.0f} ms  "
          f"end2end {batch / total_s:7.0f} steps/s  return(last20) {statistics.fmean(ep_ret[-20:]) if ep_ret else float('nan'):.1f}", flush=True)


def run(ctx: RunContext) -> None:
    for hidden in ctx.exp_cfg["hidden_sizes"]:
        try:
            run_config(ctx, hidden)
        except Exception as exc:
            ctx.record_failure(exc, model=f"ppo_mlp{hidden}", dataset=ctx.exp_cfg["env"], precision="fp32", hidden=hidden)
        ctx.cleanup()


def main(argv=None) -> int:
    return experiment_main(EXPERIMENT, run, description="PPO / CartPole benchmark", argv=argv)


if __name__ == "__main__":
    sys.exit(main())
