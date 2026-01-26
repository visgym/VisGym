import numpy as np
import ast 
import gymnasium as gym
from gymnasium import spaces
from typing import NamedTuple, Optional, Dict
from PIL import Image, ImageDraw, ImageFont
from gymnasium.envs.matchstick_equation import matchstick_puzzles
from gymnasium.spaces import FuncConditional


class MatchstickEquationState(NamedTuple):
    broken: list                # CURRENT equation symbols
    history: list               # stack of prior symbol-lists
    done: bool
    max_steps: int # The maximum number of actions allowed to solve the equation


class MatchstickEquationEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array", "ansi"]}

    def __init__(
        self,
        render_mode="rgb_array",
        seed: int = 42,
        max_steps=1000,
        image_width=300,
        image_height=80,
        break_moves=1,
        enforce_min_distance: bool = True,  # ← NEW: require minimal repair ≥ break_moves
        gen_retry_limit: int = 300, # ← NEW: cap retries during generation
    ):
        """
        :param render_mode:       Must be "rgb_array" or None for this example
        :param max_steps:         Episode step limit
        :param image_width:       For rendering the puzzle as an image
        :param image_height:      For rendering the puzzle as an image
        """
        self.render_mode = render_mode or "rgb_array"
        self.max_steps = max_steps
        self.image_width = image_width
        self.image_height = image_height
        self.rng = np.random.default_rng(seed)
        self.break_moves = int(break_moves)
        self.current_moves = 0
        self.enforce_min_distance = enforce_min_distance
        self.gen_retry_limit = gen_retry_limit
        self.max_equation_len = 30
        self.segment_ids = sorted(set(matchstick_puzzles.TOTAL))
        self.max_segments = (max(self.segment_ids) + 1)  # e.g., 13 for ids 0..12

        self.action_space = FuncConditional({
            "move": spaces.MultiDiscrete([
                self.max_equation_len,  # src_idx
                self.max_segments,      # src_seg  (now 0..12 ok)
                self.max_equation_len,  # dst_idx
                self.max_segments       # dst_seg
            ]),
            "undo": spaces.Text(4),     # must be 'undo'
            "stop": spaces.Text(4)      # must be 'stop'
        })

        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(self.image_height, self.image_width, 3),
            dtype=np.uint8
        )
        
        self.SEGMENTS_TO_SYMBOL = {
            frozenset(segs): sym
            for sym, segs in matchstick_puzzles.SYMBOL_TO_SEGMENTS.items()
        }
        try:
            self.font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        except:
            try:
                self.font = ImageFont.load_default()
                self.font_size = 16
            except:
                self.font = None
                self.font_size = 16

        self.state: MatchstickEquationState = None
    
    def seed(self, seed: Optional[int] = None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]
        

             
    def get_prompt(self, **kwargs) -> str:
        tokens = getattr(self, "tokens", None)
        N = len(tokens) if tokens is not None else kwargs.get("num_symbols", None)

        # Start with the basic task description
        prompt = (
            "You see a broken matchstick equation.\n"
            "Your goal is to fix the equation by moving ONE match per action."
        )
        if self.render_mode == 'ansi':
            prompt += f"\nYou see an ASCII visualization in text for the equation. A single match could be represented by two or three characters depending on its orientation."
        else: # rgb_array
            prompt += "\nYou see an image of the equation."
        
        prompt += "\n\nSymbols are indexed 0..N-1 from left to right (N = number of symbols)."

        prompt += "\n\nAvailable actions:\n"
        action_descriptions = []
        actions = self.action_space.get_function_names()
        
        if "move" in actions:
            action_descriptions.append(
                f"{len(action_descriptions) + 1}. 'move': Remove one match from segment 'src_seg' of symbol at 'src_idx', "
                "then add it to segment 'dst_seg' of symbol at 'dst_idx'. "
                "Format: `('move', [src_idx, src_seg, dst_idx, dst_seg])` where:\n"
                "   • src_idx, dst_idx ∈ [0, N−1], with src_idx ≠ dst_idx\n"
                "   • src_seg, dst_seg ∈ matchstick_puzzles.TOTAL (e.g., {0..12})\n"
                "   • The move must result in valid symbols at BOTH positions."
            )
        if "undo" in actions:
            action_descriptions.append(
                f"{len(action_descriptions) + 1}. 'undo': Revert the last move (if any). "
                "Format: `('undo', 'undo')`"
            )
        if "stop" in actions:
            action_descriptions.append(
                f"{len(action_descriptions) + 1}. 'stop': Submit your current equation as final. "
                "Format: `('stop', 'stop')`"
            )
        
        prompt += "\n".join(action_descriptions)
        prompt += (
            "\n\nSuccess: The submitted equation must be mathematically correct (evaluated as lhs == rhs)."
        )
        prompt += (
            "\n\nSegment legend (indices depend on symbol):\n"
            "  0..6 : 7-seg digits (a,b,c,d,e,f,g), 6 is also the horizontal stroke used by '+'\n"
            "  7    : plus vertical stroke (used by '+')\n"
            "  8    : the multiply sign that goes from top left to bottom right\n" # the multiply sign that goes from top left to bottom right
            "  9    : the multiply sign that goes from top right to bottom left\n"
            # "  10   : diagonal for divide (if enabled in mapping)\n"
            "  11,12: equals upper/lower bars (used by '=')\n"
            "A segment is valid for a symbol only if the resulting set of segments maps to a known glyph."
        )

        prompt += "\n\nPlease respond with exactly one action and its arguments in the specified format. For example:\n"
        examples = []
        if "move" in actions:
            examples.append("- To move a match: `('move', [0, 6, 2, 0])`")
        if "undo" in actions:
            examples.append("- To undo: `('undo', 'undo')`")
        if "stop" in actions:
            examples.append("- To stop: `('stop', 'stop')`")
        
        prompt += "\n".join(examples)
        return prompt

    def get_init_state(self) -> Dict:
        """
        Get the current environment state for reproducibility.
        Returns a dict that can be passed to reset(init_state=...) to recreate the same environment.
        Note: This should be called after reset() to capture the current state.
        """
        initial_broken = self.state.history[0] if self.state.history else self.state.broken
        return {
            'broken': initial_broken.copy() if isinstance(initial_broken, list) else list(initial_broken),
        }
    
    def reset(self, seed=None, options=None, init_state: Optional[Dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        if init_state is not None:
            broken = init_state['broken'].copy()
        else:
            broken = self._create_puzzle(self.break_moves)
        
        self.current_moves = 0
        self.tokens = broken[:]  # useful for prompt

        self.state = MatchstickEquationState(
            broken=broken,
            history=[broken],           
            max_steps=self.max_steps,
            done=False,
        )

        obs = self._get_obs()
        return obs, self._get_info()

    def step(self, action):
        terminated = truncated = False
        reward = 0.0
        self.env_feedback = None

        try:
            parsed = ast.literal_eval(action)
        except Exception as exc:
            self.env_feedback = f"Cannot parse action string {action!r}: {exc} so the action is invalid."
            return self._get_obs(), 0.0, terminated, truncated, self._get_info()
        
        if (not isinstance(parsed, tuple)) or len(parsed) != 2:
            self.env_feedback = f"The action format is invalid. The action should be a tuple of two elements with the first element being the action name and the second element being the action payload."
            return self._get_obs(), 0.0, terminated, truncated, self._get_info()
        
        branch, payload = parsed[0], parsed[1]

        # 3. Execute ------------------------------------------------------
        if branch == "move":
            if not self.action_space["move"].contains(payload):
                self.env_feedback = f"The action {payload} is not in the valid action space of {branch} so nothing is executed."
            elif self.current_moves >= self.break_moves:
                self.env_feedback = "Move budget exhausted. Please try to undo."
            else:
                # Try to apply the move
                if self._apply_move(payload):
                    self.current_moves += 1
                    self.env_feedback = None
                else:
                    self.env_feedback = "Illegal move."
        
        elif branch == "undo":
            if not self.action_space["undo"].contains(payload):
                self.env_feedback = f"The action {payload} is not in the valid action space of {branch} so nothing is executed."
            elif self._apply_undo():
                self.env_feedback = None
            else:
                self.env_feedback = "Nothing to undo."

        elif branch == "stop":
            if not self.action_space["stop"].contains(payload):
                self.env_feedback = f"The action {payload} is not in the valid action space of {branch} so nothing is executed."
            else:
                terminated = True
                truncated = False
        else:
            self.env_feedback = f"The action name '{branch}' is not recognized in the available action space."
        
        # 4. Compute reward ------------------------------------------------------
        if terminated:
            reward = self._compute_reward()
        
        # 5. Return ------------------------------------------------------
        if self.env_feedback is None:
            self.env_feedback = "Action executed successfully."
        info = self._get_info()
        observation = self._get_obs()
        return observation, reward, terminated, truncated, info

    def render(self, mode = "rgb_array"):
        """
        Compute the render frames as specified by `mode`.
        """
        if mode == "ansi":
            return self._render_ansi()
        elif mode == "rgb_array":
            return self._render_frame()
        else:
            raise ValueError(f"Invalid render mode: {mode}")

    def solve(
        self,
        strategy: str = "sos",
        num_steps: int = None,
        max_moves: int = None,
        sos_seed: int = None,
        sos_max_len: int = None,
        validate: bool = True,
    ):
        if max_moves is None:
            max_moves = int(self.break_moves)

        start = tuple(self.state.broken)
        if matchstick_puzzles.is_equation_correct("".join(start)):
            return [str(('stop','stop'))]

        # --- 1) Golden path via BFS (also used by SoS) ---------------------
        from collections import deque
        def shortest_fix_actions(src_state, cap):
            q = deque([(src_state, [])])
            seen = {src_state}
            while q:
                curr, path = q.popleft()
                if matchstick_puzzles.is_equation_correct("".join(curr)):
                    return path  # list of ('move', [i,s,j,t])
                if len(path) >= cap:
                    continue
                for nxt, act in self._neighbors_one_move(curr):
                    if nxt in seen:
                        continue
                    seen.add(nxt)
                    q.append((nxt, path + [act]))
            return None

        golden = shortest_fix_actions(start, max_moves)
        if golden is None:
            return []  # unsolvable within budget (shouldn't happen if generation is consistent)

        if strategy.lower() == "bfs":
            stream = [str(a) for a in golden] + [str(('stop','stop'))]
            if not validate:
                return stream
            ok, why = self._dry_rollout_ok(stream)
            return stream if ok else []

        elif strategy.lower() == "dfs":
            # produce a full backtracking trace with undos
            visited = {start}
            out = []

            def dfs(state, depth):
                if matchstick_puzzles.is_equation_correct("".join(state)):
                    return True
                if depth >= max_moves:
                    return False
                for nxt, act in self._neighbors_one_move(state):
                    if nxt in visited:
                        continue
                    visited.add(nxt)
                    out.append(act)
                    if dfs(nxt, depth + 1):
                        return True
                    out.append(('undo','undo'))
                return False

            found = dfs(start, 0)
            if not found:
                return []
            stream = [str(a) for a in out] + [str(('stop','stop'))]
            if not validate:
                return stream
            ok, why = self._dry_rollout_ok(stream)
            if ok:
                return stream
            # Fallback to golden if the long trace fails on action-space constraints
            return [str(a) for a in golden] + [str(('stop','stop'))]

        elif strategy.lower() == "sos":
            # --- 2) SoS: add reversible detours without breaking budgets ----
            if sos_seed is None:
                sos_seed = int(self.rng.integers(0, 2**32))
            local_rng = np.random.default_rng(sos_seed)

            G = len(golden)
            # Total tokens emitted = G moves + 2*D (detour ops & undos) + 1 stop
            detour_ops_budget = 0
            if sos_max_len is not None:
                detour_ops_budget = max(0, (sos_max_len - 1 - G) // 2)
            else:
                detour_ops_budget = 2 * G  # a mild default

            stream_actions = []
            cur_state = start

            # Precompute states along the golden path to avoid stepping into them on detour
            golden_states = [start]
            s = start
            for act in golden:
                s = self._apply_action_to_state_once(s, act[1])
                golden_states.append(s)

            for k, golden_act in enumerate(golden):
                # remaining move budget *before* taking this golden step
                slack_moves = int(self.break_moves) - k - 1
                if detour_ops_budget > 0 and slack_moves > 0:
                    # how many ops can we spend on this detour
                    per_cap = min(2, slack_moves, detour_ops_budget)  # <=2 by default; tweak if you like
                    detour_depth = int(local_rng.integers(0, per_cap + 1))
                    detour_state = cur_state
                    ops_done = 0
                    for _ in range(detour_depth):
                        nbrs = self._neighbors_one_move(detour_state)
                        if not nbrs:
                            break
                        local_rng.shuffle(nbrs)
                        # avoid immediately jumping to the next golden state to keep variety
                        avoid = golden_states[k + 1]
                        cands = [p for p in nbrs if p[0] != avoid] or nbrs
                        nxt_state, nxt_act = cands[0]
                        # Ensure the detour move stays in action space (paranoia after the fix)
                        if not self.action_space["move"].contains(nxt_act[1]):
                            continue
                        stream_actions.append(nxt_act)
                        detour_state = nxt_state
                        ops_done += 1
                    # rewind with undos
                    stream_actions.extend([('undo','undo')] * ops_done)
                    detour_ops_budget -= ops_done

                # take the golden step
                stream_actions.append(golden_act)
                cur_state = self._apply_action_to_state_once(cur_state, golden_act[1])

            stream = [str(a) for a in stream_actions] + [str(('stop','stop'))]
            if not validate:
                return stream
            ok, why = self._dry_rollout_ok(stream)
            if ok:
                return stream
            # Fallback: pure golden path
            return [str(a) for a in golden] + [str(('stop','stop'))]

        else:
            raise ValueError("Unknown strategy: choose 'bfs', 'dfs', or 'sos'")

    def close(self):
        pass

    def _get_obs(self):
        return self._render_frame()

    def _get_info(self):
        eq_str = "".join(self.state.broken)
        return {
            "equation": eq_str,
            "is_correct": matchstick_puzzles.is_equation_correct(eq_str),
            "max_steps": self.state.max_steps,
            "history_depth": len(self.state.history),
            "current_moves": self.current_moves,
            "env_feedback": getattr(self, "env_feedback", None)
        }

    def _compute_reward(self):
        eq_str = "".join(self.state.broken)
        correct = matchstick_puzzles.is_equation_correct(eq_str)
        return 1.0 if correct else 0.0

########################################################
# End of all methods, start of helper methods
########################################################

    def _render_ansi(self):
        """
        Render the environment to an ANSI string, drawing each symbol with ASCII characters.
        """
        # ASCII art for each segment on a 5x5 grid
        SEGMENT_ASCII = {
            # 7-segment display parts
            0: [(0, 1, "---")],
            1: [(1, 4, "|"), (2, 4, "|")],
            2: [(3, 4, "|"), (4, 4, "|")],
            3: [(5, 1, "---")],
            4: [(3, 0, "|"), (4, 0, "|")],
            5: [(1, 0, "|"), (2, 0, "|")],
            6: [(2, 1, "---")],
            # Special segments
            7: [(1, 2, "|"), (2, 2, "|"), (3, 2, "|")],  # Vertical for '+'
            8: [(1, 1, "\\"), (2, 2, "\\"), (3, 3, "\\")],  # Multiply '\'
            9: [(1, 3, "/"), (2, 2, "/"), (3, 1, "/")],  # Multiply '/'
            10: [(1, 3, "/"), (2, 2, "/"), (3, 1, "/")], # Divide '/'
            11: [(1, 1, "---")],  # Equals top
            12: [(4, 1, "---")],  # Equals bottom
        }
        
        symbols = self.state.broken
        num_symbols = len(symbols)
        symbol_width = 5
        symbol_height = 6  # 5 for symbol, 1 for padding
        canvas_height = symbol_height + 2 # +2 for index labels
        canvas_width = num_symbols * (symbol_width + 1)

        # Initialize canvas with spaces
        canvas = [[' ' for _ in range(canvas_width)] for _ in range(canvas_height)]

        for i, char_symbol in enumerate(symbols):
            if char_symbol not in matchstick_puzzles.SYMBOL_TO_SEGMENTS:
                continue
            
            segments = matchstick_puzzles.SYMBOL_TO_SEGMENTS[char_symbol]
            char_x_offset = i * (symbol_width + 1)

            # Draw segments for the current symbol
            for seg_id in segments:
                if seg_id in SEGMENT_ASCII:
                    for r, c, char in SEGMENT_ASCII[seg_id]:
                        for k, ch in enumerate(char):
                            if 0 <= r < symbol_height and 0 <= char_x_offset + c + k < canvas_width:
                                canvas[r][char_x_offset + c + k] = ch
            
            # Add index label
            index_label = str(i)
            label_x = char_x_offset + (symbol_width - len(index_label)) // 2
            for k, ch in enumerate(index_label):
                 if 0 <= label_x + k < canvas_width:
                    canvas[symbol_height][label_x + k] = ch

        # Convert canvas to a single string
        return "\n".join("".join(row) for row in canvas)

    def _render_frame(self):
        """
        Render the current puzzle (self.state.broken) as an RGB array.
        """
        eq_str = "".join(self.state.broken)
        img = self._draw_equation_image(eq_str)
        img = img.resize((self.image_width, self.image_height))
        return np.array(img, dtype=np.uint8)

    def _generate_random_equation(self):
        """Randomly choose one template and return a correct equation string (without spaces)."""
        TEMPLATES = [self._eq_template_add, self._eq_template_sub, self._eq_template_mul,
                self._eq_template_div, self._eq_template_mixed]
        template = self.rng.choice(TEMPLATES)
        return template().replace(" ", "")

    def _create_puzzle(self, break_moves: int):
        """
        Start from a correct equation and apply EXACTLY `break_moves` one-match moves
        to produce a WRONG equation. If enforce_min_distance=True, ensure minimal
        repair distance is ≥ break_moves (reject easier ones).
        """
        SYM2SEGS = matchstick_puzzles.SYMBOL_TO_SEGMENTS
        SEGS2SYM = self.SEGMENTS_TO_SYMBOL
        ALL_SEGS = set(matchstick_puzzles.TOTAL)

        for _retry in range(self.gen_retry_limit):
            correct_eq = self._generate_random_equation()
            eq_list = list(correct_eq)
            n = len(eq_list)
            if n > self.max_equation_len:
                continue

            ok = True
            # apply exactly `break_moves` random valid one-match moves
            for _k in range(break_moves):
                moved = False
                for _tries in range(200):
                    i = int(self.rng.integers(0, n))
                    j = int(self.rng.integers(0, n))
                    if i == j:
                        continue
                    si, sj = eq_list[i], eq_list[j]
                    if si not in SYM2SEGS or sj not in SYM2SEGS:
                        continue

                    src_segs = set(SYM2SEGS[si])
                    dst_segs = set(SYM2SEGS[sj])
                    if not src_segs:
                        continue

                    s_choices = list(src_segs); self.rng.shuffle(s_choices)
                    t_choices = list(ALL_SEGS - dst_segs); self.rng.shuffle(t_choices)

                    done_one = False
                    for s in s_choices:
                        new_src = src_segs.copy(); new_src.remove(s)
                        new_src_sym = SEGS2SYM.get(frozenset(new_src))
                        if new_src_sym is None:
                            continue
                        for t in t_choices:
                            new_dst = dst_segs.copy(); new_dst.add(t)
                            new_dst_sym = SEGS2SYM.get(frozenset(new_dst))
                            if new_dst_sym is None:
                                continue
                            eq_list[i], eq_list[j] = new_src_sym, new_dst_sym
                            moved = True
                            done_one = True
                            break
                        if done_one:
                            break
                    if moved:
                        break
                if not moved:
                    ok = False
                    break

            if not ok:
                continue

            broken = eq_list
            broken_str = "".join(broken)

            # Must be WRONG
            if matchstick_puzzles.is_equation_correct(broken_str):
                continue

            # Enforce minimal distance ≥ break_moves (reject if solvable in < break_moves)
            if self.enforce_min_distance:
                d = self._min_fix_distance(broken, cap=break_moves)
                if d is not None:
                    # Found a fix in < break_moves; reject and retry
                    continue

            return broken

        # Fallback if generation fails repeatedly: degrade to 1-move breaker
        print("Fallback if generation fails repeatedly: degrade to 1-move breaker")
        for _ in range(100):
            correct_eq = self._generate_random_equation()
            eq_list = list(correct_eq)
            n = len(eq_list)
            if n > self.max_equation_len:
                continue
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    si, sj = eq_list[i], eq_list[j]
                    if si in SYM2SEGS and sj in SYM2SEGS:
                        for (new_src, new_dst) in matchstick_puzzles.possible_match_moves(si, sj, k=1):
                            old_si, old_sj = eq_list[i], eq_list[j]
                            eq_list[i], eq_list[j] = new_src, new_dst
                            if not matchstick_puzzles.is_equation_correct("".join(eq_list)):
                                return eq_list
                            eq_list[i], eq_list[j] = old_si, old_sj
        return list(self._generate_random_equation())

    def _eq_template_add(self):
        A = int(self.rng.integers(0, 100))
        B = int(self.rng.integers(0, 100))
        return f"{A}+{B}={A+B}"

    def _eq_template_sub(self):
        A = int(self.rng.integers(0, 100))
        B = int(self.rng.integers(0, A+1))
        return f"{A}-{B}={A-B}"

    def _eq_template_mul(self):
        A = int(self.rng.integers(0, 21))
        B = int(self.rng.integers(0, 21))
        return f"{A}*{B}={A*B}"

    def _eq_template_div(self):
        B = int(self.rng.integers(1, 21))
        C = int(self.rng.integers(0, 21))
        A = B * C
        return f"{A}/{B}={C}"

    def _eq_template_mixed(self):
        A = int(self.rng.integers(0, 21))
        B = int(self.rng.integers(0, 21))
        C = int(self.rng.integers(0, 11))
        return f"{A}+{B}*{C}={A+B*C}"

    def _neighbors_one_move(self, sym_tuple):
        """
        Generate all states reachable by exactly ONE valid ('move', [i,s,j,t]) move
        from the given tuple of symbols. Yields (next_tuple, action_tuple).
        """
        SYM2SEGS = matchstick_puzzles.SYMBOL_TO_SEGMENTS
        SEGS2SYM = self.SEGMENTS_TO_SYMBOL
        ALL_SEGS = set(matchstick_puzzles.TOTAL)

        N = len(sym_tuple)
        out = []
        for i in range(N):
            si = sym_tuple[i]
            if si not in SYM2SEGS:
                continue
            src_segs = set(SYM2SEGS[si])
            if not src_segs:
                continue

            for s in list(src_segs):
                new_src_set = src_segs.copy(); new_src_set.remove(s)
                new_src_sym = SEGS2SYM.get(frozenset(new_src_set))
                if new_src_sym is None:
                    continue

                for j in range(N):
                    if j == i:
                        continue
                    sj = sym_tuple[j]
                    if sj not in SYM2SEGS:
                        continue
                    dst_segs = set(SYM2SEGS[sj])

                    for t in (ALL_SEGS - dst_segs):
                        new_dst_set = dst_segs.copy(); new_dst_set.add(t)
                        new_dst_sym = SEGS2SYM.get(frozenset(new_dst_set))
                        if new_dst_sym is None:
                            continue

                        nxt = list(sym_tuple)
                        nxt[i] = new_src_sym
                        nxt[j] = new_dst_sym
                        out.append((tuple(nxt), ('move', [i, s, j, t])))
        return out

    def _min_fix_distance(self, start_syms, cap: int):
        """
        Return the minimal number of one-match moves needed to reach a CORRECT equation,
        **strictly less than `cap`**. If no solution exists in < cap moves, return None.

        Used to enforce that the generated broken puzzle is not solvable in < break_moves.
        """
        start = tuple(start_syms)
        # If already correct (shouldn't be), distance is 0
        if matchstick_puzzles.is_equation_correct("".join(start)):
            return 0

        from collections import deque
        q = deque([(start, 0)])
        seen = {start}

        while q:
            state, d = q.popleft()
            if d >= cap:          # we only care about solutions with depth < cap
                continue

            for nxt, _ in self._neighbors_one_move(state):
                if nxt in seen:
                    continue
                nd = d + 1
                # Found a solution in < cap moves?
                if nd < cap and matchstick_puzzles.is_equation_correct("".join(nxt)):
                    return nd
                seen.add(nxt)
                if nd < cap:
                    q.append((nxt, nd))

        return None

    def _apply_move(self, payload):
        """Apply a move action and return True if successful, False otherwise."""
        try:
            i, s, j, t = map(int, payload)
        except Exception:
            return False
        
        eq_now = list(self.state.broken)
        
        # Check indices and symbols are valid
        if not (0 <= i < len(eq_now) and 0 <= j < len(eq_now) and i != j and
                eq_now[i] in matchstick_puzzles.SYMBOL_TO_SEGMENTS and
                eq_now[j] in matchstick_puzzles.SYMBOL_TO_SEGMENTS):
            return False
        
        src_segs = set(matchstick_puzzles.SYMBOL_TO_SEGMENTS[eq_now[i]])
        dst_segs = set(matchstick_puzzles.SYMBOL_TO_SEGMENTS[eq_now[j]])
        
        # Check if move is valid
        if not (s in src_segs and t not in dst_segs):
            return False
        
        # Apply the move
        new_src = src_segs - {s}
        new_dst = dst_segs | {t}
        sym_src = self.SEGMENTS_TO_SYMBOL.get(frozenset(new_src))
        sym_dst = self.SEGMENTS_TO_SYMBOL.get(frozenset(new_dst))
        
        if sym_src is None or sym_dst is None:
            return False
        
        # Update the equation
        eq_now[i], eq_now[j] = sym_src, sym_dst
        
        # Update state
        history = list(self.state.history)
        history.append(eq_now.copy())
        self.state = MatchstickEquationState(
            broken=eq_now,
            history=history,
            done=False,
            max_steps=self.state.max_steps,
        )
        
        return True

    def _apply_undo(self):
        """Apply an undo action and return True if successful, False otherwise."""
        history = list(self.state.history)
        if len(history) <= 1:
            return False
        
        self.current_moves -= 1
        history.pop()
        eq_now = history[-1].copy()
        
        self.state = MatchstickEquationState(
            broken=eq_now,
            history=history,
            done=False,
            max_steps=self.state.max_steps,
        )
        
        return True

    def _draw_equation_image(self, eq_str):
        """
        Draws the equation with per-symbol index labels underneath each symbol.
        """
        symbols = [ch for ch in eq_str]
        n = len(symbols)
        symbol_w = 60
        symbol_h = 80
        padding_h = 20  # Extra space for index labels
        symbol_spacing = 8

        # img_w = n * symbol_w
        img_w = n * symbol_w + (n - 1) * symbol_spacing
        img_h = symbol_h + padding_h
        img = Image.new("RGB", (img_w, img_h), (255, 255, 255))
        draw_obj = ImageDraw.Draw(img)

        x_cursor = 0
        for idx, ch in enumerate(symbols):
            if ch in matchstick_puzzles.SYMBOL_TO_SEGMENTS:
                self._draw_symbol_realistic(draw_obj, ch, (x_cursor, 0))
            text = str(idx)
            if self.font is not None:
                bbox = draw_obj.textbbox((0, 0), text, font=self.font)
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]
                text_x = x_cursor + (symbol_w - text_w) // 2
                text_y = symbol_h + (padding_h - text_h) // 2
                draw_obj.text((text_x, text_y), text, fill=(0, 0, 0), font=self.font)
            else:
                # Fallback: draw text without font (will use default)
                text_w = len(text) * 8  # rough estimate
                text_h = 12  # rough estimate
                text_x = x_cursor + (symbol_w - text_w) // 2
                text_y = symbol_h + (padding_h - text_h) // 2
                draw_obj.text((text_x, text_y), text, fill=(0, 0, 0))
            # x_cursor += symbol_w
            x_cursor += symbol_w + symbol_spacing
        return img


    def _draw_symbol_realistic(self, draw, symbol, offset=(0, 0),
                               thickness=4, head_radius=4):
        """
        Minimal inline version of draw_symbol_realistic that draws the symbol's segments.
        """
        from math import hypot, atan2, sin, cos
        from PIL import ImageDraw

        # Use your global dictionary
        SEGMENT_COORDS = {
            0: ((10, 5), (30, 5)),
            1: ((30, 5), (30, 25)),
            2: ((30, 25), (30, 45)),
            3: ((10, 45), (30, 45)),
            4: ((10, 25), (10, 45)),
            5: ((10, 5), (10, 25)),
            6: ((10, 25), (30, 25)),
            7: ((20, 5), (20, 45)),
            8: ((10, 10), (30, 40)),
            9: ((30, 10), (10, 40)),
            10: ((10, 40), (30, 10)),
            11: ((10, 15), (30, 15)),
            12: ((10, 35), (30, 35)),
        }
        symbol_segments = matchstick_puzzles.SYMBOL_TO_SEGMENTS[symbol]
        ox, oy = offset
        for seg_id in symbol_segments:
            (sx, sy), (ex, ey) = SEGMENT_COORDS[seg_id]
            start = (ox + sx, oy + sy)
            end = (ox + ex, oy + ey)
            self._draw_realistic_match(draw, start, end, 
                            match_color=(235,210,150),
                            outline_color=(100,80,40),
                            head_color=(200,0,0),
                            thickness=6,
                            head_radius=6,
                            tail_radius=4)


    def _draw_realistic_match(
        self,
        draw,
        start,
        end,
        match_color=(235, 210, 150),
        outline_color=(100,  80,  40),
        head_color=(200,   0,   0),
        thickness=6,
        head_radius=6,
        tail_radius=4
    ):
        """
        Draw one matchstick from `start` to `end`:
        - thick line with round caps, outlined in dark brown
        - red head at `start`
        - small wood‐colored cap at `end`
        """
        x1, y1 = start
        x2, y2 = end

        # 1) Outline (slightly thicker, darker) so bodies never visually merge
        draw.line([start, end], fill=outline_color, width=thickness+2)

        # 2) Wood body (centered over the outline)
        draw.line([start, end], fill=match_color, width=thickness)

        # 3) Red match head at the 'start' point
        r = head_radius
        draw.ellipse((x1-r, y1-r, x1+r, y1+r), fill=head_color)

        # 4) Small tail‐cap at the 'end' point
        tr = tail_radius
        draw.ellipse((x2-tr, y2-tr, x2+tr, y2+tr), fill=match_color)

    def _apply_action_to_state_once(self, state_tuple, payload):
        """Pure transition: state_tuple -> tuple or None for illegal."""
        i, s, j, t = map(int, payload)
        toks = list(state_tuple)
        if not (0 <= i < len(toks) and 0 <= j < len(toks) and i != j):
            return None
        SYM2SEGS = matchstick_puzzles.SYMBOL_TO_SEGMENTS
        if toks[i] not in SYM2SEGS or toks[j] not in SYM2SEGS:
            return None
        src = set(SYM2SEGS[toks[i]]); dst = set(SYM2SEGS[toks[j]])
        if s not in src or t in dst:
            return None
        new_src = src - {s}; new_dst = dst | {t}
        sym_src = self.SEGMENTS_TO_SYMBOL.get(frozenset(new_src))
        sym_dst = self.SEGMENTS_TO_SYMBOL.get(frozenset(new_dst))
        if sym_src is None or sym_dst is None:
            return None
        out = list(toks)
        out[i], out[j] = sym_src, sym_dst
        return tuple(out)

    def _dry_rollout_ok(self, stream):
        """Validate a stream of action *strings* without touching env state."""
        import ast
        stack = [tuple(self.state.broken)]
        moves = 0
        for s in stream:
            branch, payload = ast.literal_eval(s)
            if branch == 'move':
                if moves >= int(self.break_moves):
                    return False, "move-budget-exhausted"
                nxt = self._apply_action_to_state_once(stack[-1], payload)
                if nxt is None:
                    return False, "illegal-move"
                stack.append(nxt)
                moves += 1
            elif branch == 'undo':
                if len(stack) == 1:
                    return False, "undo-with-empty-history"
                stack.pop()
                moves = max(0, moves-1)
            elif branch == 'stop':
                break
            else:
                return False, f"unknown-branch:{branch}"
        final_ok = matchstick_puzzles.is_equation_correct("".join(stack[-1]))
        return (final_ok, "ok" if final_ok else "final-not-correct")