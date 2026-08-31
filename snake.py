"""
Step 1: the game. No brain yet.

Arcade Snake. Grid, discrete steps, four-direction movement, walls kill.

Two things are non-standard, and only two:

  * The action space is RELATIVE -- turn left, go straight, turn right. Not four
    absolute directions. This costs nothing (reversing is illegal in Snake
    anyway) and it is what a descending-neuron readout naturally produces:
    three outputs, argmax.

  * The observation is EGOCENTRIC -- the agent sees the world from its own head,
    rotated so "up" is whichever way it is currently moving. It never gets a
    top-down board. A fly brain has no machinery for allocentric maps; it does
    have machinery for "what is in front of me".

Human mode still uses arrow keys as absolute directions, because that is how
Snake feels right to play. They get converted to relative internally.

Run it:
    python snake.py --mode human
    python snake.py --mode bot
    python snake.py --mode bot --headless --episodes 40
    python snake.py --mode human --obs retina     # see the egocentric patch

Controls: arrows or WASD to steer, A/D also work as turn-left/turn-right,
R restarts, ESC quits, SPACE pauses.
"""

import argparse
from collections import deque
import numpy as np

# Absolute directions, indexed 0..3. Turning right = +1, left = -1 (mod 4).
DIRS = [(1, 0), (0, 1), (-1, 0), (0, -1)]   # E, N, W, S
LEFT, STRAIGHT, RIGHT = 0, 1, 2
ACTION_TURN = {LEFT: -1, STRAIGHT: 0, RIGHT: +1}

FEATURE_NAMES = [
    "danger_L", "danger_F", "danger_R",       # one cell away
    "danger_L2", "danger_F2", "danger_R2",    # two cells away
    "food_sin", "food_cos",                   # food bearing, relative to heading
    "food_near",                              # 1 - normalised distance
    "fullness",                               # length / board area
]


class SnakeEnv:
    """Single agent. Step 4 vectorises this over a population."""

    def __init__(self, width=20, height=20, wrap=False, obs="feature",
                 retina=7, max_idle=200, seed=None):
        self.W = width
        self.H = height
        self.wrap = wrap            # False = walls kill, the classic behaviour
        self.obs_mode = obs
        self.K = retina             # side length of the egocentric patch
        self.max_idle = max_idle    # starve if you go this long without eating
        self.rng = np.random.default_rng(seed)
        self.reset()

    @property
    def n_obs(self):
        return len(FEATURE_NAMES) if self.obs_mode == "feature" \
            else self.K * self.K * 3

    # ------------------------------------------------------------------ reset

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        cx, cy = self.W // 2, self.H // 2
        self.dir = int(self.rng.integers(0, 4))
        dx, dy = DIRS[self.dir]
        # start length 3, laid out behind the head
        self.body = deque([(cx - i * dx, cy - i * dy) for i in range(3)])
        self.occupied = set(self.body)
        self.score = 0
        self.steps = 0
        self.idle = 0
        self.alive = True
        self._place_food()
        return self.observe()

    def _place_food(self):
        free = [(x, y) for x in range(self.W) for y in range(self.H)
                if (x, y) not in self.occupied]
        if not free:                       # board solved
            self.food = None
            self.alive = False
            return
        self.food = free[int(self.rng.integers(0, len(free)))]

    # ------------------------------------------------------------- perception

    def _ahead(self, turn, n=1):
        """Cell n steps in a direction relative to the current heading."""
        d = DIRS[(self.dir + turn) % 4]
        hx, hy = self.body[0]
        x, y = hx + d[0] * n, hy + d[1] * n
        return (x % self.W, y % self.H) if self.wrap else (x, y)

    def _is_deadly(self, cell):
        x, y = cell
        if not self.wrap and not (0 <= x < self.W and 0 <= y < self.H):
            return True
        return cell in self.occupied

    def observe(self):
        if self.obs_mode == "retina":
            return self._observe_retina()

        hx, hy = self.body[0]
        f = [float(self._is_deadly(self._ahead(t, n)))
             for n in (1, 2) for t in (-1, 0, 1)]

        # food bearing in the snake's own frame
        if self.food is None:
            fx = fy = 0.0
        else:
            dx, dy = self.food[0] - hx, self.food[1] - hy
            if self.wrap:                  # take the short way round
                dx = (dx + self.W // 2) % self.W - self.W // 2
                dy = (dy + self.H // 2) % self.H - self.H // 2
            # rotate world delta into heading-relative coords
            ang = self.dir * (np.pi / 2)
            ca, sa = np.cos(-ang), np.sin(-ang)
            fx, fy = dx * ca - dy * sa, dx * sa + dy * ca
        r = np.hypot(fx, fy)
        bearing = [fy / r, fx / r] if r > 0 else [0.0, 1.0]  # sin, cos
        near = 1.0 - min(r / (self.W + self.H), 1.0)
        fullness = len(self.body) / (self.W * self.H)

        return np.array(f + bearing + [near, fullness], dtype=np.float64)

    def _observe_retina(self):
        """K x K patch centred on the head, rotated so the heading points up.

        Three channels: body, food, wall. This is the observation to use with a
        connectome -- it is retinotopic, so it drops straight into a visual
        input population instead of an abstract feature vector.
        """
        K, half = self.K, self.K // 2
        out = np.zeros((3, K, K))
        hx, hy = self.body[0]
        ang = self.dir * (np.pi / 2)
        ca, sa = np.cos(ang), np.sin(ang)
        for j in range(K):          # j = forward offset (row 0 = farthest ahead)
            for i in range(K):      # i = lateral offset
                lx, ly = i - half, half - j
                # Rotate local (right, forward) into world coords. Local forward
                # must map onto the heading itself, and local right onto the
                # heading turned clockwise -- getting this backwards renders the
                # body sideways instead of trailing behind the head.
                wx = int(round(hx + lx * sa + ly * ca))
                wy = int(round(hy - lx * ca + ly * sa))
                if self.wrap:
                    wx, wy = wx % self.W, wy % self.H
                elif not (0 <= wx < self.W and 0 <= wy < self.H):
                    out[2, j, i] = 1.0
                    continue
                if (wx, wy) in self.occupied:
                    out[0, j, i] = 1.0
                if self.food == (wx, wy):
                    out[1, j, i] = 1.0
        return out.ravel()

    # -------------------------------------------------------------------- step

    def step(self, action):
        """action in {0, 1, 2} = turn left, straight, turn right."""
        if not self.alive:
            return self.observe(), 0.0, True

        self.dir = (self.dir + ACTION_TURN[int(action)]) % 4
        head = self._ahead(0, 1)

        reward = -0.01                       # small cost per step, discourage stalling
        tail = self.body[-1]
        growing = (self.food is not None and head == self.food)

        # the tail cell vacates this tick, so following it is legal
        deadly = self._is_deadly(head) and not (head == tail and not growing)
        if deadly:
            self.alive = False
            return self.observe(), reward - 10.0, True

        self.body.appendleft(head)
        self.occupied.add(head)
        if growing:
            self.score += 1
            self.idle = 0
            reward += 10.0
            self._place_food()
        else:
            self.body.pop()
            self.occupied.discard(tail)
            self.idle += 1

        self.steps += 1
        if self.idle >= self.max_idle:       # went too long without eating
            self.alive = False
            return self.observe(), reward - 5.0, True
        return self.observe(), reward, not self.alive


# --------------------------------------------------------------------- baseline

def greedy_bot(env):
    """Move toward the food, refuse to die this tick. That is the whole policy.

    Your floor. It is deliberately shortsighted -- it will happily wall itself
    into a pocket -- so there is real headroom for a policy with memory.
    """
    hx, hy = env.body[0]
    best, best_score = STRAIGHT, -1e9
    for a in (LEFT, STRAIGHT, RIGHT):
        cell = env._ahead(ACTION_TURN[a], 1)
        if env._is_deadly(cell):
            continue
        if env.food is None:
            score = 0.0
        else:
            dx, dy = env.food[0] - cell[0], env.food[1] - cell[1]
            if env.wrap:
                dx = (dx + env.W // 2) % env.W - env.W // 2
                dy = (dy + env.H // 2) % env.H - env.H // 2
            score = -(abs(dx) + abs(dy))
        score += 0.1 if a == STRAIGHT else 0.0     # tiebreak: keep going
        if score > best_score:
            best, best_score = a, score
    return best


# ----------------------------------------------------------------------- render

COLORS = {
    "bg": (16, 18, 24), "grid": (26, 30, 40), "head": (255, 214, 102),
    "body": (90, 200, 160), "food": (255, 110, 120), "text": (200, 210, 225),
    "dim": (110, 120, 140), "hud": (24, 27, 36), "bar": (120, 200, 255),
}


def play(mode="human", seed=None, headless=False, episodes=1, fps=12,
         obs="feature", width=20, height=20, wrap=False, retina=7):
    if headless:
        scores = []
        for ep in range(episodes):
            env = SnakeEnv(width, height, wrap, obs, retina,
                           seed=None if seed is None else seed + ep)
            done = False
            while not done:
                _, _, done = env.step(greedy_bot(env))
            scores.append(env.score)
            print(f"episode {ep:3d}  score {env.score:3d}  steps {env.steps}")
        print(f"\nmean {np.mean(scores):.2f}  max {max(scores)}  min {min(scores)}")
        return scores

    import pygame
    pygame.init()
    CELL, HUD_H, PAD = 24, 150, 12
    env = SnakeEnv(width, height, wrap, obs, retina, seed=seed)
    BW, BH = env.W * CELL, env.H * CELL
    screen = pygame.display.set_mode((BW + 2 * PAD, BH + 2 * PAD + HUD_H))
    pygame.display.set_caption(f"snake [{mode}]")
    font = pygame.font.SysFont("monospace", 13)
    big = pygame.font.SysFont("monospace", 20, bold=True)
    clock = pygame.time.Clock()

    def cell_rect(c):
        x, y = c
        return pygame.Rect(PAD + x * CELL + 1,
                           PAD + (env.H - 1 - y) * CELL + 1,
                           CELL - 2, CELL - 2)

    pending, paused, running = None, False, True
    observation = env.observe()

    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    running = False
                elif e.key == pygame.K_r:
                    observation, pending = env.reset(), None
                elif e.key == pygame.K_SPACE:
                    paused = not paused
                # arrows / WASD = absolute direction, the classic feel.
                # Converted to a relative turn below; reversing is ignored.
                elif e.key in (pygame.K_UP, pygame.K_w):
                    pending = 1
                elif e.key in (pygame.K_LEFT,):
                    pending = 2
                elif e.key in (pygame.K_DOWN, pygame.K_s):
                    pending = 3
                elif e.key in (pygame.K_RIGHT,):
                    pending = 0
                elif e.key == pygame.K_a:
                    pending = (env.dir + 1) % 4     # turn left
                elif e.key == pygame.K_d:
                    pending = (env.dir - 1) % 4     # turn right

        if env.alive and not paused:
            if mode == "bot":
                action = greedy_bot(env)
            elif pending is None:
                action = STRAIGHT
            else:
                diff = (pending - env.dir) % 4
                action = {0: STRAIGHT, 1: LEFT, 3: RIGHT}.get(diff, STRAIGHT)
                pending = None
            observation, _, _ = env.step(action)

        screen.fill(COLORS["bg"])
        for x in range(env.W + 1):
            pygame.draw.line(screen, COLORS["grid"], (PAD + x * CELL, PAD),
                             (PAD + x * CELL, PAD + BH))
        for y in range(env.H + 1):
            pygame.draw.line(screen, COLORS["grid"], (PAD, PAD + y * CELL),
                             (PAD + BW, PAD + y * CELL))
        if env.food:
            pygame.draw.rect(screen, COLORS["food"], cell_rect(env.food),
                             border_radius=6)
        for i, c in enumerate(env.body):
            col = COLORS["head"] if i == 0 else COLORS["body"]
            pygame.draw.rect(screen, col, cell_rect(c), border_radius=4)

        # ---- HUD: exactly what the brain will receive
        hud_y = PAD * 2 + BH
        pygame.draw.rect(screen, COLORS["hud"], (0, hud_y, BW + 2 * PAD, HUD_H))
        if env.obs_mode == "feature":
            for i, (name, v) in enumerate(zip(FEATURE_NAMES, observation)):
                col_i, row_i = i // 5, i % 5
                x0, y = PAD + col_i * 240, hud_y + 8 + row_i * 17
                screen.blit(font.render(f"{name:>9}", True, COLORS["text"]), (x0, y))
                pygame.draw.rect(screen, COLORS["bar"],
                                 (x0 + 78, y + 3, max(int(abs(v) * 90), 1), 9))
                screen.blit(font.render(f"{v:+.2f}", True, COLORS["dim"]),
                            (x0 + 175, y))
        else:
            patch = observation.reshape(3, env.K, env.K)
            px = PAD
            for ch, label in enumerate(["body", "food", "wall"]):
                screen.blit(font.render(label, True, COLORS["text"]),
                            (px, hud_y + 6))
                for j in range(env.K):
                    for i in range(env.K):
                        v = patch[ch, j, i]
                        pygame.draw.rect(
                            screen,
                            tuple(int(c * (0.15 + 0.85 * v)) for c in COLORS["bar"]),
                            (px + i * 14, hud_y + 24 + j * 14, 12, 12))
                px += env.K * 14 + 24
            screen.blit(font.render("egocentric, heading = up", True, COLORS["dim"]),
                        (px, hud_y + 24))

        status = f"score {env.score}  len {len(env.body)}"
        if not env.alive:
            status += "  DEAD - R"
        elif paused:
            status += "  PAUSED"
        screen.blit(big.render(status, True, COLORS["text"]),
                    (BW + 2 * PAD - 260, hud_y + HUD_H - 28))
        pygame.display.flip()
        clock.tick(fps)

    pygame.quit()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="human", choices=["human", "bot"])
    ap.add_argument("--obs", default="feature", choices=["feature", "retina"])
    ap.add_argument("--width", type=int, default=20)
    ap.add_argument("--height", type=int, default=20)
    ap.add_argument("--retina", type=int, default=7)
    ap.add_argument("--wrap", action="store_true", help="edges wrap instead of killing")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--fps", type=int, default=12)
    a = ap.parse_args()
    play(a.mode, a.seed, a.headless, a.episodes, a.fps, a.obs,
         a.width, a.height, a.wrap, a.retina)