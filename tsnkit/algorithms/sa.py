import math
import multiprocessing
import random
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, TimeoutError, as_completed
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .. import core as utils


@dataclass
class EvalSummary:
    cost: float
    scheduled_count: int
    total_delay: int
    total_offset: int
    total_hops: int
    failed_streams: List[utils.Stream]
    failed_positions: Dict[utils.Stream, int]


@dataclass
class EliteState:
    cost: float
    order: List[utils.Stream]
    path_choice: Dict[utils.Stream, int]
    failed_streams: List[utils.Stream]

@dataclass(frozen=True)
class PeriodicInterval:
    start: int
    end: int
    period: int

    @property
    def length(self) -> int:
        return self.end - self.start


class PeriodicOccupancy:
    def __init__(self) -> None:
        self.intervals: List[PeriodicInterval] = []

    def __len__(self) -> int:
        return len(self.intervals)

    def add(self, interval: PeriodicInterval) -> None:
        self.intervals.append(interval)

    def remove(self, interval: PeriodicInterval) -> None:
        for index in range(len(self.intervals) - 1, -1, -1):
            if self.intervals[index] == interval:
                self.intervals.pop(index)
                return

    def find_overlap(self, interval: PeriodicInterval) -> Optional[PeriodicInterval]:
        for occupied in self.intervals:
            if sa.periodic_intervals_overlap(interval, occupied):
                return occupied
        return None

def benchmark(
    name, task_path, net_path, output_path="./", workers=1
) -> utils.Statistics:
    stat = utils.Statistics(name)
    try:
        test = sa(workers)
        test.init(task_path, net_path)
        test.prepare()
        stat = test.solve()
        if stat.result == utils.Result.schedulable:
            test.output().to_csv(name, output_path)
        stat.content(name=name)
        return stat
    except KeyboardInterrupt:
        stat.content(name=name)
        return stat
    except Exception as e:
        print("[!]", e, flush=True)
        traceback.print_exc()
        stat.result = utils.Result.error
        stat.content(name=name)
        return stat


def _run_sa_worker(
    task_path: str,
    net_path: str,
    seed: int,
    deadline_wall: float,
) -> Tuple[utils.Statistics, float, Optional[utils.Config]]:
    worker = sa(workers=1, seed=seed, deadline_wall=deadline_wall)
    worker.init(task_path, net_path)
    worker.prepare()
    stat = worker.solve()
    if stat.result == utils.Result.schedulable:
        return stat, worker.best_cost, worker.output()
    return stat, worker.best_cost, None


class sa:
    def __init__(
        self,
        workers=1,
        seed: Optional[int] = None,
        deadline_wall: Optional[float] = None,
    ) -> None:
        self.workers = workers
        self.seed = utils.SEED if seed is None else seed
        self.deadline_wall = deadline_wall
        self.max_candidate_paths = 4
        self.max_iterations = 0
        self.initial_temperature = 1.0
        self.cooling_rate = 0.995
        self.min_temperature = 1e-3
        self.restart_period = 0
        self.stagnation_limit = 0
        self.use_incremental_eval = False
        self.post_feasible_patience = 0
        self.elite_pool_size = 4
        self.max_focus_streams = 4
        self.large_instance_mode = False
        self.feasible_iteration_cap = 0
        self.rng = random.Random(self.seed)

        self.best_cost = math.inf
        self.best_routing_map: Dict[utils.Stream, utils.Path] = {}
        self.best_offset_map: Dict[utils.Stream, int] = {}
        self.best_delay_map: Dict[utils.Stream, int] = {}
        self.best_queue_map: Dict[Tuple[utils.Stream, utils.Link], int] = {}
        self.failure_score: Dict[utils.Stream, int] = {}
        self.elite_states: List[EliteState] = []
        self.parallel_output_config: Optional[utils.Config] = None

    def init(self, task_path: str, net_path: str) -> None:
        self.task_path = task_path
        self.net_path = net_path
        self.parallel_output_config = None
        self.task = utils.load_stream(task_path)
        self.net = utils.load_network(net_path)
        stream_count = len(self.task.streams)

        if stream_count >= 160:
            self.max_candidate_paths = 2
        elif stream_count >= 96:
            self.max_candidate_paths = 3
        else:
            self.max_candidate_paths = 4

        self.task_routes = {
            stream: self.select_candidate_paths(stream) for stream in self.task.streams
        }
        self.unscheduled_penalty = max(
            10000.0,
            float(sum(stream.deadline for stream in self.task.streams)) * 2.0,
        )
        self.large_instance_mode = stream_count >= 96
        if stream_count <= 16:
            self.max_iterations = max(220, stream_count * 20)
            self.feasible_iteration_cap = max(120, stream_count * 24)
        elif stream_count <= 32:
            self.max_iterations = max(500, stream_count * 40)
            self.feasible_iteration_cap = max(220, stream_count * 20)
        elif stream_count <= 96:
            self.max_iterations = max(900, stream_count * 24)
            self.feasible_iteration_cap = max(320, stream_count * 14)
        elif stream_count <= 160:
            self.max_iterations = max(900, stream_count * 12)
            self.feasible_iteration_cap = max(420, stream_count * 8)
        else:
            self.max_iterations = max(1000, stream_count * 8)
            self.feasible_iteration_cap = max(480, stream_count * 6)

        self.initial_temperature = max(5.0, float(stream_count) * (1.2 if self.large_instance_mode else 2.0))
        self.restart_period = max(32, stream_count * (2 if self.large_instance_mode else 6))
        self.stagnation_limit = max(64, stream_count * (3 if self.large_instance_mode else 10))
        self.use_incremental_eval = stream_count >= 24
        self.post_feasible_patience = (
            max(6, min(24, stream_count // 8))
            if self.large_instance_mode
            else max(8, stream_count * 2)
        )
        self.failure_score = {stream: 0 for stream in self.task.streams}
        self.elite_states = []

        self.task_order = sorted(
            self.task.streams,
            key=lambda stream: (
                self.get_min_path_delay(stream),
                stream.deadline,
                stream.period,
                -len(self.task_routes[stream]),
                int(stream),
            ),
        )
        self.initial_path_choice = {stream: 0 for stream in self.task.streams}
        self.reset_schedule_state()

    def prepare(self) -> None:
        pass

    def is_deadline_reached(self, start_time: float) -> bool:
        if self.deadline_wall is not None:
            return time.monotonic() >= self.deadline_wall
        return utils.time_log() - start_time > utils.T_LIMIT

    def solve_parallel(self) -> utils.Statistics:
        start_time = utils.time_log()
        deadline_wall = self.deadline_wall
        if deadline_wall is None:
            deadline_wall = time.monotonic() + utils.T_LIMIT

        worker_count = max(1, self.workers)
        ctx = multiprocessing.get_context("spawn")
        best_stat: Optional[utils.Statistics] = None
        best_cost = math.inf
        best_config: Optional[utils.Config] = None

        with ProcessPoolExecutor(max_workers=worker_count, mp_context=ctx) as executor:
            futures = [
                executor.submit(
                    _run_sa_worker,
                    self.task_path,
                    self.net_path,
                    self.seed + index * 9973,
                    deadline_wall,
                )
                for index in range(worker_count)
            ]

            remaining = max(0.0, deadline_wall - time.monotonic())
            try:
                for future in as_completed(futures, timeout=remaining + 1.0):
                    stat, worker_cost, config = future.result()
                    if stat.result == utils.Result.schedulable and config is not None:
                        if worker_cost < best_cost:
                            best_cost = worker_cost
                            best_stat = stat
                            best_config = config
                    elif best_stat is None:
                        best_stat = stat
            except TimeoutError:
                pass

        run_time = utils.time_log() - start_time
        if best_config is not None and best_stat is not None:
            self.best_cost = best_cost
            self.parallel_output_config = best_config
            return utils.Statistics("-", utils.Result.schedulable, run_time)
        if best_stat is not None:
            return utils.Statistics("-", best_stat.result, run_time)
        return utils.Statistics("-", utils.Result.unknown, run_time)

    @utils.check_time_limit
    def solve(self) -> utils.Statistics:
        if self.workers > 1:
            return self.solve_parallel()

        return self._solve_single()

    def _solve_single(self) -> utils.Statistics:
        self.parallel_output_config = None
        self.best_cost = math.inf
        self.best_routing_map = {}
        self.best_offset_map = {}
        self.best_delay_map = {}
        self.best_queue_map = {}
        start_time = utils.time_log()
        for stream in self.task.streams:
            if self.get_min_path_delay(stream) > stream.deadline:
                return utils.Statistics(
                    "-", utils.Result.unschedulable, utils.time_log() - start_time
                )

        current_order = list(self.task_order)
        current_paths = dict(self.initial_path_choice)
        current_eval = self.evaluate_state(current_order, current_paths)

        best_partial_order = list(current_order)
        best_partial_paths = dict(current_paths)
        best_partial_eval = current_eval
        stagnation_steps = 0
        hard_restart_count = 0
        first_feasible_step: Optional[int] = None

        self.record_elite_state(current_order, current_paths, current_eval)
        self.update_failure_history(current_eval.failed_streams)

        if current_eval.scheduled_count == len(self.task.streams):
            self.capture_best_schedule(current_eval.cost)
            first_feasible_step = 0

        temperature = self.initial_temperature
        for step in range(self.max_iterations):
            if self.is_deadline_reached(start_time):
                break

            neighbor_order, neighbor_paths = self.make_neighbor(
                current_order, current_paths, current_eval.failed_streams
            )
            if self.use_incremental_eval:
                neighbor_eval, first_affected_index = self.evaluate_neighbor(
                    current_order,
                    current_paths,
                    current_eval,
                    neighbor_order,
                    neighbor_paths,
                )
            else:
                neighbor_eval = self.evaluate_state(neighbor_order, neighbor_paths)
            self.update_failure_history(neighbor_eval.failed_streams)
            self.record_elite_state(neighbor_order, neighbor_paths, neighbor_eval)

            if neighbor_eval.cost < best_partial_eval.cost:
                best_partial_order = list(neighbor_order)
                best_partial_paths = dict(neighbor_paths)
                best_partial_eval = neighbor_eval
                stagnation_steps = 0
            else:
                stagnation_steps += 1

            if neighbor_eval.scheduled_count == len(self.task.streams):
                if neighbor_eval.cost < self.best_cost:
                    self.capture_best_schedule(neighbor_eval.cost)
                    stagnation_steps = 0
                    if first_feasible_step is None:
                        first_feasible_step = step

            delta = neighbor_eval.cost - current_eval.cost
            if delta <= 0 or self.accept_worse(delta, temperature):
                current_order = neighbor_order
                current_paths = neighbor_paths
                current_eval = neighbor_eval
            else:
                if self.use_incremental_eval:
                    self.restore_base_suffix(
                        current_order,
                        current_paths,
                        neighbor_order,
                        first_affected_index,
                    )
                else:
                    self.evaluate_state(current_order, current_paths)

            temperature *= self.cooling_rate
            periodic_restart = (step + 1) % self.restart_period == 0
            if temperature < self.min_temperature or periodic_restart or stagnation_steps >= self.stagnation_limit:
                hard_restart = temperature < self.min_temperature or stagnation_steps >= self.stagnation_limit
                current_order, current_paths, current_eval = self.restart_search(
                    best_partial_order,
                    best_partial_paths,
                    best_partial_eval,
                    current_order,
                    current_paths,
                    current_eval,
                    hard_restart=hard_restart,
                )
                self.record_elite_state(current_order, current_paths, current_eval)
                temperature = self.reset_temperature(hard_restart, hard_restart_count)
                stagnation_steps = 0
                if hard_restart:
                    hard_restart_count += 1

            if first_feasible_step is not None and step - first_feasible_step >= self.post_feasible_patience:
                break

            if self.best_cost < math.inf and step > self.feasible_iteration_cap:
                break

        run_time = utils.time_log() - start_time
        if self.best_cost < math.inf:
            self.restore_best_schedule()
            return utils.Statistics("-", utils.Result.schedulable, run_time)
        return utils.Statistics("-", utils.Result.unknown, run_time)

    def _new_link_occupancy(self) -> Dict[utils.Link, PeriodicOccupancy]:
        return {link: PeriodicOccupancy() for link in self.net.links}

    def _new_link_queue_occupancy(self) -> Dict[utils.Link, Dict[int, PeriodicOccupancy]]:
        return {
            link: {queue: PeriodicOccupancy() for queue in range(link.q_num)}
            for link in self.net.links
        }

    def reset_schedule_state(self) -> None:
        self.task_routing_map: Dict[utils.Stream, utils.Path] = {}
        self.offset_map: Dict[utils.Stream, int] = {}
        self.network_delay: Dict[utils.Stream, int] = {}
        self.queue_map: Dict[Tuple[utils.Stream, utils.Link], int] = {}
        self.link_occupancy = self._new_link_occupancy()
        self.link_queue_occupancy = self._new_link_queue_occupancy()

    def evaluate_state(
        self, order: List[utils.Stream], path_choice: Dict[utils.Stream, int]
    ) -> EvalSummary:
        self.reset_schedule_state()
        return self.schedule_suffix(order, path_choice, 0, None)

    def evaluate_neighbor(
        self,
        base_order: List[utils.Stream],
        base_paths: Dict[utils.Stream, int],
        base_eval: EvalSummary,
        order: List[utils.Stream],
        path_choice: Dict[utils.Stream, int],
    ) -> Tuple[EvalSummary, int]:
        first_affected_index = self.get_first_affected_index(
            base_order, base_paths, order, path_choice
        )
        if first_affected_index >= len(order):
            return base_eval, first_affected_index

        self.remove_scheduled_streams(base_order[first_affected_index:])
        prefix_summary = self.summarize_prefix(base_order, first_affected_index)
        return (
            self.schedule_suffix(order, path_choice, first_affected_index, prefix_summary),
            first_affected_index,
        )

    def summarize_prefix(
        self, order: List[utils.Stream], prefix_end: int
    ) -> EvalSummary:
        scheduled_count = 0
        total_delay = 0
        total_offset = 0
        total_hops = 0
        failed_streams: List[utils.Stream] = []
        failed_positions: Dict[utils.Stream, int] = {}

        for position, stream in enumerate(order[:prefix_end]):
            if stream in self.task_routing_map:
                scheduled_count += 1
                total_delay += self.network_delay[stream]
                total_offset += self.offset_map[stream]
                total_hops += len(self.task_routing_map[stream].links)
            else:
                failed_streams.append(stream)
                failed_positions[stream] = position

        return self.build_summary(
            scheduled_count,
            total_delay,
            total_offset,
            total_hops,
            failed_streams,
            failed_positions,
        )

    def schedule_suffix(
        self,
        order: List[utils.Stream],
        path_choice: Dict[utils.Stream, int],
        start_index: int,
        prefix_summary: Optional[EvalSummary],
    ) -> EvalSummary:
        if prefix_summary is None:
            scheduled_count = 0
            total_delay = 0
            total_offset = 0
            total_hops = 0
            failed_streams: List[utils.Stream] = []
            failed_positions: Dict[utils.Stream, int] = {}
        else:
            scheduled_count = prefix_summary.scheduled_count
            total_delay = prefix_summary.total_delay
            total_offset = prefix_summary.total_offset
            total_hops = prefix_summary.total_hops
            failed_streams = list(prefix_summary.failed_streams)
            failed_positions = dict(prefix_summary.failed_positions)

        for position in range(start_index, len(order)):
            stream = order[position]
            routes = self.task_routes[stream]
            if not routes:
                failed_streams.append(stream)
                failed_positions[stream] = position
                continue

            route_index = path_choice.get(stream, 0) % len(routes)
            path = routes[route_index]
            delay = self.get_nw_delay_for_path(stream, path)
            if delay > stream.deadline:
                failed_streams.append(stream)
                failed_positions[stream] = position
                continue

            offset, queue_assignment = self.find_inject_offset_for_path(stream, path)
            if offset < 0 or queue_assignment is None:
                failed_streams.append(stream)
                failed_positions[stream] = position
                continue

            self.assign_stream(stream, path, offset, delay, queue_assignment)
            scheduled_count += 1
            total_delay += delay
            total_offset += offset
            total_hops += len(path.links)

        return self.build_summary(
            scheduled_count,
            total_delay,
            total_offset,
            total_hops,
            failed_streams,
            failed_positions,
        )

    def build_summary(
        self,
        scheduled_count: int,
        total_delay: int,
        total_offset: int,
        total_hops: int,
        failed_streams: List[utils.Stream],
        failed_positions: Dict[utils.Stream, int],
    ) -> EvalSummary:
        unscheduled = len(self.task.streams) - scheduled_count
        cost = (
            unscheduled * self.unscheduled_penalty
            + total_delay
            + 0.1 * total_offset
            + 0.5 * total_hops
        )
        return EvalSummary(
            cost=cost,
            scheduled_count=scheduled_count,
            total_delay=total_delay,
            total_offset=total_offset,
            total_hops=total_hops,
            failed_streams=failed_streams,
            failed_positions=failed_positions,
        )

    def restore_base_suffix(
        self,
        base_order: List[utils.Stream],
        base_paths: Dict[utils.Stream, int],
        neighbor_order: List[utils.Stream],
        first_affected_index: int,
    ) -> None:
        if first_affected_index >= len(base_order):
            return
        self.remove_scheduled_streams(neighbor_order[first_affected_index:])
        prefix_summary = self.summarize_prefix(base_order, first_affected_index)
        self.schedule_suffix(base_order, base_paths, first_affected_index, prefix_summary)

    def remove_scheduled_streams(self, streams: List[utils.Stream]) -> None:
        for stream in reversed(streams):
            self.unassign_stream(stream)

    def unassign_stream(self, stream: utils.Stream) -> None:
        path = self.task_routing_map.get(stream)
        offset = self.offset_map.get(stream)
        if path is None or offset is None:
            return

        prev_end = offset
        for link in path.links:
            transmit = stream.get_t_trans(link)
            start = prev_end
            end = start + transmit
            interval = PeriodicInterval(start, end, stream.period)
            self.link_occupancy[link].remove(interval)
            queue = self.queue_map[(stream, link)]
            self.link_queue_occupancy[link][queue].remove(interval)
            prev_end = end + link.t_proc

        for link in path.links:
            self.queue_map.pop((stream, link), None)
        self.task_routing_map.pop(stream, None)
        self.offset_map.pop(stream, None)
        self.network_delay.pop(stream, None)

    def get_first_affected_index(
        self,
        base_order: List[utils.Stream],
        base_paths: Dict[utils.Stream, int],
        order: List[utils.Stream],
        path_choice: Dict[utils.Stream, int],
    ) -> int:
        for index, (left, right) in enumerate(zip(base_order, order)):
            if left != right:
                return index
        changed_positions = [
            index
            for index, stream in enumerate(order)
            if self.task_routes[stream]
            and (base_paths.get(stream, 0) % len(self.task_routes[stream]))
            != (path_choice.get(stream, 0) % len(self.task_routes[stream]))
        ]
        if changed_positions:
            return min(changed_positions)
        return len(order)

    def make_neighbor(
        self,
        order: List[utils.Stream],
        path_choice: Dict[utils.Stream, int],
        failed_streams: List[utils.Stream],
    ) -> Tuple[List[utils.Stream], Dict[utils.Stream, int]]:
        next_order = list(order)
        next_paths = dict(path_choice)

        focus_streams = self.get_focus_streams(failed_streams)

        if self.large_instance_mode:
            action = self.rng.random()
            if action < 0.55:
                self.mutate_paths(next_paths, focus_streams, max_mutations=1)
            elif action < 0.82 and len(next_order) > 1:
                self.promote_streams(
                    next_order,
                    focus_streams,
                    front_window=max(1, len(next_order) // 5),
                )
            else:
                self.local_swap(next_order, focus_streams)
            return next_order, next_paths

        action = self.rng.random()
        if action < 0.20 and len(next_order) > 1:
            left, right = sorted(self.rng.sample(range(len(next_order)), 2))
            next_order[left], next_order[right] = next_order[right], next_order[left]
        elif action < 0.40 and len(next_order) > 1:
            self.promote_streams(next_order, focus_streams, front_window=max(1, len(next_order) // 3))
        elif action < 0.57 and len(next_order) > 3:
            self.move_block(next_order, focus_streams)
        elif action < 0.72 and len(next_order) > 3:
            self.reverse_subsequence(next_order, focus_streams)
        elif action < 0.87:
            self.mutate_paths(next_paths, focus_streams, max_mutations=2)
        else:
            self.promote_streams(next_order, focus_streams, front_window=max(1, len(next_order) // 4))
            self.mutate_paths(next_paths, focus_streams, max_mutations=3)

        return next_order, next_paths

    def local_swap(self, order: List[utils.Stream], focus_streams: List[utils.Stream]) -> None:
        if len(order) <= 1:
            return
        if focus_streams and focus_streams[0] in order:
            anchor_index = order.index(focus_streams[0])
        else:
            anchor_index = self.rng.randrange(len(order))
        radius = min(6, len(order) - 1)
        left = max(0, anchor_index - radius)
        right = min(len(order) - 1, anchor_index + radius)
        if left == right:
            return
        swap_index = self.rng.randint(left, right)
        if swap_index == anchor_index:
            swap_index = left if swap_index < right else right
        order[anchor_index], order[swap_index] = order[swap_index], order[anchor_index]

    def get_focus_streams(self, failed_streams: List[utils.Stream]) -> List[utils.Stream]:
        if failed_streams:
            ordered_failed = sorted(
                failed_streams,
                key=lambda stream: (-self.failure_score.get(stream, 0), int(stream)),
            )
            return ordered_failed[: self.max_focus_streams]

        ranked_streams = sorted(
            self.task.streams,
            key=lambda stream: (-self.failure_score.get(stream, 0), int(stream)),
        )
        top_ranked = [stream for stream in ranked_streams if self.failure_score.get(stream, 0) > 0]
        if top_ranked:
            return top_ranked[: self.max_focus_streams]
        return self.rng.sample(self.task.streams, min(self.max_focus_streams, len(self.task.streams)))

    def promote_streams(
        self, order: List[utils.Stream], streams: List[utils.Stream], front_window: int
    ) -> None:
        if len(order) <= 1:
            return
        insert_limit = min(front_window, len(order) - 1)
        for stream in streams:
            if stream not in order:
                continue
            source_index = order.index(stream)
            target_index = self.rng.randrange(insert_limit + 1)
            item = order.pop(source_index)
            order.insert(target_index, item)

    def move_block(self, order: List[utils.Stream], focus_streams: List[utils.Stream]) -> None:
        if len(order) <= 3:
            return
        if focus_streams:
            anchor_index = order.index(focus_streams[0])
        else:
            anchor_index = self.rng.randrange(len(order))
        block_radius = min(2, len(order) // 4)
        start = max(0, anchor_index - self.rng.randint(0, block_radius))
        end = min(len(order), anchor_index + self.rng.randint(1, block_radius + 1) + 1)
        block = order[start:end]
        del order[start:end]
        target_index = self.rng.randrange(len(order) + 1)
        order[target_index:target_index] = block

    def reverse_subsequence(self, order: List[utils.Stream], focus_streams: List[utils.Stream]) -> None:
        if len(order) <= 3:
            return
        if focus_streams:
            anchor_index = order.index(focus_streams[0])
            left = max(0, anchor_index - self.rng.randint(1, min(3, anchor_index + 1)))
            right = min(
                len(order) - 1,
                anchor_index + self.rng.randint(1, min(3, len(order) - anchor_index)),
            )
        else:
            left, right = sorted(self.rng.sample(range(len(order)), 2))
        order[left : right + 1] = reversed(order[left : right + 1])

    def mutate_paths(
        self,
        path_choice: Dict[utils.Stream, int],
        focus_streams: List[utils.Stream],
        max_mutations: int,
    ) -> None:
        candidates = list(focus_streams)
        if len(candidates) < max_mutations:
            for stream in self.task.streams:
                if stream not in candidates:
                    candidates.append(stream)
        mutation_count = min(max_mutations, len(candidates))
        chosen_streams = self.rng.sample(candidates, mutation_count)
        for stream in chosen_streams:
            num_paths = len(self.task_routes[stream])
            if num_paths <= 1:
                continue
            current_index = path_choice.get(stream, 0) % num_paths
            preferred_indexes = [index for index in range(min(3, num_paths)) if index != current_index]
            all_indexes = [index for index in range(num_paths) if index != current_index]
            if preferred_indexes and self.rng.random() < 0.75:
                path_choice[stream] = self.rng.choice(preferred_indexes)
            else:
                path_choice[stream] = self.rng.choice(all_indexes)

    def update_failure_history(self, failed_streams: List[utils.Stream]) -> None:
        for stream in set(failed_streams):
            self.failure_score[stream] = self.failure_score.get(stream, 0) + 1

    def record_elite_state(
        self,
        order: List[utils.Stream],
        path_choice: Dict[utils.Stream, int],
        summary: EvalSummary,
    ) -> None:
        candidate = EliteState(
            cost=summary.cost,
            order=list(order),
            path_choice=dict(path_choice),
            failed_streams=list(summary.failed_streams),
        )
        existing_index = None
        signature = tuple(int(stream) for stream in candidate.order)
        for index, elite in enumerate(self.elite_states):
            elite_signature = tuple(int(stream) for stream in elite.order)
            if elite_signature == signature:
                existing_index = index
                break
        if existing_index is not None:
            if candidate.cost < self.elite_states[existing_index].cost:
                self.elite_states[existing_index] = candidate
        else:
            self.elite_states.append(candidate)
        self.elite_states.sort(key=lambda elite: elite.cost)
        self.elite_states = self.elite_states[: self.elite_pool_size]

    def select_restart_anchor(
        self,
        best_partial_order: List[utils.Stream],
        best_partial_paths: Dict[utils.Stream, int],
        best_partial_eval: EvalSummary,
        current_order: List[utils.Stream],
        current_paths: Dict[utils.Stream, int],
        current_eval: EvalSummary,
        hard_restart: bool,
    ) -> Tuple[List[utils.Stream], Dict[utils.Stream, int], List[utils.Stream]]:
        if not hard_restart and self.elite_states and self.rng.random() < 0.7:
            elite_index = min(len(self.elite_states) - 1, self.rng.randrange(min(3, len(self.elite_states))))
            elite = self.elite_states[elite_index]
            return list(elite.order), dict(elite.path_choice), list(elite.failed_streams)
        if hard_restart and self.elite_states and self.rng.random() < 0.5:
            elite = self.elite_states[0]
            return list(elite.order), dict(elite.path_choice), list(elite.failed_streams)
        if best_partial_eval.cost <= current_eval.cost:
            return list(best_partial_order), dict(best_partial_paths), list(best_partial_eval.failed_streams)
        return list(current_order), dict(current_paths), list(current_eval.failed_streams)

    def restart_search(
        self,
        best_partial_order: List[utils.Stream],
        best_partial_paths: Dict[utils.Stream, int],
        best_partial_eval: EvalSummary,
        current_order: List[utils.Stream],
        current_paths: Dict[utils.Stream, int],
        current_eval: EvalSummary,
        hard_restart: bool,
    ) -> Tuple[List[utils.Stream], Dict[utils.Stream, int], EvalSummary]:
        restart_order, restart_paths, restart_failed = self.select_restart_anchor(
            best_partial_order,
            best_partial_paths,
            best_partial_eval,
            current_order,
            current_paths,
            current_eval,
            hard_restart,
        )
        focus_streams = self.get_focus_streams(restart_failed)

        if hard_restart:
            ranked_streams = sorted(
                self.task.streams,
                key=lambda stream: (-self.failure_score.get(stream, 0), stream.deadline, int(stream)),
            )
            restart_order = list(ranked_streams)
            self.mutate_paths(restart_paths, ranked_streams[: self.max_focus_streams + 1], max_mutations=4)
            self.promote_streams(restart_order, ranked_streams[: self.max_focus_streams], front_window=max(1, len(restart_order) // 4))
        else:
            self.promote_streams(restart_order, focus_streams, front_window=max(1, len(restart_order) // 4))
            self.move_block(restart_order, focus_streams)
            self.mutate_paths(restart_paths, focus_streams, max_mutations=3)

        restart_eval = self.evaluate_state(restart_order, restart_paths)
        self.update_failure_history(restart_eval.failed_streams)
        return restart_order, restart_paths, restart_eval

    def reset_temperature(self, hard_restart: bool, hard_restart_count: int) -> float:
        if hard_restart:
            scale = min(1.0, 0.65 + 0.08 * hard_restart_count)
            return max(self.initial_temperature * scale, self.min_temperature)
        return max(self.initial_temperature * 0.45, self.min_temperature)

    def accept_worse(self, delta: float, temperature: float) -> bool:
        if temperature <= 0:
            return False
        probability = math.exp(-delta / temperature)
        return self.rng.random() < probability

    def capture_best_schedule(self, cost: float) -> None:
        self.best_cost = cost
        self.best_routing_map = dict(self.task_routing_map)
        self.best_offset_map = dict(self.offset_map)
        self.best_delay_map = dict(self.network_delay)
        self.best_queue_map = dict(self.queue_map)

    def restore_best_schedule(self) -> None:
        self.task_routing_map = dict(self.best_routing_map)
        self.offset_map = dict(self.best_offset_map)
        self.network_delay = dict(self.best_delay_map)
        self.queue_map = dict(self.best_queue_map)

    def assign_stream(
        self,
        stream: utils.Stream,
        path: utils.Path,
        offset: int,
        delay: int,
        queue_assignment: Dict[utils.Link, int],
    ) -> None:
        self.task_routing_map[stream] = path
        self.offset_map[stream] = offset
        self.network_delay[stream] = delay
        for link, queue in queue_assignment.items():
            self.queue_map[(stream, link)] = queue
        self.commit_stream(stream, offset)

    def select_candidate_paths(self, stream: utils.Stream) -> List[utils.Path]:
        all_paths = self.net.get_all_path(stream.src, stream.dst)
        ranked_paths = sorted(
            all_paths,
            key=lambda path: (
                self.get_nw_delay_for_path(stream, path),
                len(path.links),
                tuple(int(link) for link in path.links),
            ),
        )
        return ranked_paths[: self.max_candidate_paths]

    def get_nw_delay_for_path(self, stream: utils.Stream, path: utils.Path) -> int:
        return sum(link.t_proc + stream.get_t_trans(link) for link in path.links)

    def get_min_path_delay(self, stream: utils.Stream) -> int:
        return min(self.get_nw_delay_for_path(stream, path) for path in self.task_routes[stream])

    def build_periodic_intervals(
        self, stream: utils.Stream, path: utils.Path, offset: int
    ) -> Dict[utils.Link, PeriodicInterval]:
        intervals: Dict[utils.Link, PeriodicInterval] = {}
        prev_end = offset
        for link in path.links:
            transmit = stream.get_t_trans(link)
            start = prev_end
            end = start + transmit
            intervals[link] = PeriodicInterval(start, end, stream.period)
            prev_end = end + link.t_proc
        return intervals

    def get_next_offset_after_conflict(
        self,
        offset: int,
        hop_offset: int,
        candidate: PeriodicInterval,
        occupied: PeriodicInterval,
    ) -> int:
        gcd_period = math.gcd(candidate.period, occupied.period)
        if gcd_period <= 1:
            return offset + 1

        candidate_start = offset + hop_offset
        for delta in range(1, gcd_period + 1):
            shifted = PeriodicInterval(
                candidate_start + delta,
                candidate_start + delta + candidate.length,
                candidate.period,
            )
            if not self.periodic_intervals_overlap(shifted, occupied):
                return offset + delta
        return offset + 1

    def find_inject_offset_for_path(
        self, stream: utils.Stream, path: utils.Path
    ) -> Tuple[int, Optional[Dict[utils.Link, int]]]:
        offset = 0
        max_offset = stream.period
        while offset < max_offset:
            feasible, queue_assignment, next_offset = self.check_offset_feasible_for_path(
                stream, path, offset
            )
            if feasible:
                return offset, queue_assignment
            offset = max(offset + 1, next_offset)
        return -1, None

    def check_offset_feasible_for_path(
        self, stream: utils.Stream, path: utils.Path, offset: int
    ) -> Tuple[bool, Optional[Dict[utils.Link, int]], int]:
        intervals_by_link = self.build_periodic_intervals(stream, path, offset)

        for link, interval in intervals_by_link.items():
            hop_offset = interval.start - offset
            occupied = self.find_first_conflict(self.link_occupancy[link], interval)
            if occupied is not None:
                next_offset = self.get_next_offset_after_conflict(
                    offset,
                    hop_offset,
                    interval,
                    occupied,
                )
                return False, None, next_offset

        queue_assignment: Dict[utils.Link, int] = {}
        for link in path.links:
            queue_candidates = sorted(
                self.link_queue_occupancy[link].items(),
                key=lambda item: (0 if len(item[1]) == 0 else 1, len(item[1]), item[0]),
            )
            assigned_queue = None
            next_offset = None
            for queue, occupied_list in queue_candidates:
                interval = intervals_by_link[link]
                occupied = self.find_first_conflict(occupied_list, interval)
                if occupied is None:
                    assigned_queue = queue
                    break
                hop_offset = interval.start - offset
                candidate_offset = self.get_next_offset_after_conflict(
                    offset,
                    hop_offset,
                    interval,
                    occupied,
                )
                if next_offset is None or candidate_offset < next_offset:
                    next_offset = candidate_offset

            if assigned_queue is None:
                return False, None, next_offset if next_offset is not None else offset + 1
            queue_assignment[link] = assigned_queue

        return True, queue_assignment, offset

    def commit_stream(self, stream: utils.Stream, offset: int) -> None:
        path = self.task_routing_map[stream]
        prev_end = offset
        for link in path.links:
            transmit = stream.get_t_trans(link)
            start = prev_end
            end = start + transmit
            interval = PeriodicInterval(start, end, stream.period)
            self.link_occupancy[link].add(interval)
            queue = self.queue_map[(stream, link)]
            self.link_queue_occupancy[link][queue].add(interval)
            prev_end = end + link.t_proc

    def output(self) -> utils.Config:
        if self.parallel_output_config is not None:
            return self.parallel_output_config
        config = utils.Config()
        config.gcl = self.get_gcl()
        config.release = self.get_offset()
        config.queue = self.get_queue()
        config.route = self.get_route()
        config._delay = self.get_delay()
        return config

    def get_gcl(self) -> utils.GCL:
        gcl = []
        for stream in self.task.streams:
            if stream not in self.offset_map:
                continue
            path = self.task_routing_map[stream]
            offset = self.offset_map[stream]
            prev_end = offset
            for link in path.links:
                start = prev_end
                end = start + stream.get_t_trans(link)
                queue = self.queue_map.get((stream, link), 0)
                for frame_index in stream.get_frame_indexes(self.task.lcm):
                    gcl.append(
                        [
                            link,
                            queue,
                            start + frame_index * stream.period,
                            end + frame_index * stream.period,
                            self.task.lcm,
                        ]
                    )
                prev_end = end + link.t_proc
        return utils.GCL(gcl)

    def get_offset(self) -> utils.Release:
        offset = []
        for stream, release in self.offset_map.items():
            offset.append([stream, 0, release])
        return utils.Release(offset)

    def get_queue(self) -> utils.Queue:
        queue = []
        for stream in self.task.streams:
            if stream not in self.task_routing_map:
                continue
            for link in self.task_routing_map[stream].links:
                queue.append([stream, 0, link, self.queue_map.get((stream, link), 0)])
        return utils.Queue(queue)

    def get_route(self) -> utils.Route:
        route = []
        for stream in self.task.streams:
            if stream not in self.task_routing_map:
                continue
            for link in self.task_routing_map[stream].links:
                route.append([stream, link])
        return utils.Route(route)

    def get_delay(self) -> utils.Delay:
        delay = []
        for stream in self.task.streams:
            if stream not in self.network_delay:
                continue
            path = self.task_routing_map[stream]
            net_delay = self.network_delay[stream] - path.links[0].t_proc - stream.get_t_trans(
                path.links[0]
            )
            delay.append([stream, 0, net_delay])
        return utils.Delay(delay)

    @staticmethod
    def periodic_intervals_overlap(
        left: PeriodicInterval, right: PeriodicInterval
    ) -> bool:
        gcd_period = math.gcd(left.period, right.period)
        residue = (right.start - left.start) % gcd_period
        return residue < left.length or (residue != 0 and gcd_period - residue < right.length)

    @staticmethod
    def find_first_conflict(
        occupied: PeriodicOccupancy, interval: PeriodicInterval
    ) -> Optional[PeriodicInterval]:
        return occupied.find_overlap(interval)


if __name__ == "__main__":
    args = utils.parse_command_line_args()
    utils.Statistics().header()
    benchmark(args.name, args.task, args.net, args.output, args.workers)