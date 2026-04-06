import traceback
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import gurobipy as gp

from .. import core as utils


Occupancy = Tuple[int, int, utils.Stream]
AssignmentMap = Dict[utils.Stream, "Assignment"]


@dataclass(frozen=True)
class Assignment:
    path: utils.Path
    offset: int
    queue_map: Dict[utils.Link, int]
    hop_starts: Dict[utils.Link, int]
    hop_ends: Dict[utils.Link, int]
    delay: int
    score: float


@dataclass(frozen=True)
class LocalRepairOption:
    path_index: int
    offset: int
    assignment: Assignment


def benchmark(
    name, task_path, net_path, output_path="./", workers=1
) -> utils.Statistics:
    stat = utils.Statistics(name)
    try:
        test = jrs_bwq(workers)  # type: ignore
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
    except Exception as exc:
        print("[!]", exc, flush=True)
        traceback.print_exc()
        stat.result = utils.Result.error
        stat.content(name=name)
        return stat


class jrs_bwq:
    def __init__(self, workers=1) -> None:
        self.workers = workers
        self.max_candidate_paths = 4
        self.offset_scan_points = 12
        self.max_offset_trials = 32
        self.max_wait = 4
        self.repair_path_budget = 2
        self.repair_neighborhood = 8
        self.local_branch_limit = 3
        self.local_search_limit = 80
        self.max_schedule_attempts = 3
        self.time_check_interval = 16
        self.gurobi_repair_candidate_limit = 5
        self.gurobi_repair_time_limit = 2.0

    def init(self, task_path: str, net_path: str) -> None:
        self.task = utils.load_stream(task_path)
        self.net = utils.load_network(net_path)
        self.is_prepared = False

    def prepare(self) -> None:
        self.initialize_precomputed_state()
        self.reset_runtime_state()
        self.is_prepared = True

    def initialize_precomputed_state(self) -> None:
        self.stream_frames: Dict[utils.Stream, List[int]] = {
            stream: list(stream.get_frame_indexes(self.task.lcm)) for stream in self.task
        }

        self.task_routes = {
            stream: self.net.get_all_path(stream.src, stream.dst) for stream in self.task
        }
        self.path_delay: Dict[Tuple[utils.Stream, Tuple[utils.Link, ...]], int] = {}
        self.path_static_score: Dict[Tuple[utils.Stream, Tuple[utils.Link, ...]], float] = {}
        self.candidate_paths: Dict[utils.Stream, List[utils.Path]] = {}
        for stream in self.task:
            ranked_paths = self.rank_paths(stream)
            self.candidate_paths[stream] = ranked_paths[: self.max_candidate_paths]
        self.min_path_delay: Dict[utils.Stream, int] = {
            stream: min(
                self.get_nw_delay_for_path(stream, path)
                for path in self.candidate_paths[stream]
            )
            for stream in self.task
        }
        self.stream_priority: Dict[utils.Stream, float] = {
            stream: self.get_stream_priority(stream) for stream in self.task
        }

        self.base_stream_order = sorted(
            self.task.streams,
            key=lambda stream: self.stream_priority[stream],
            reverse=True,
        )
        self.stream_orders = self.build_stream_orders()

    def reset_runtime_state(self) -> None:
        self.assignments: AssignmentMap = {}
        self.assignment_entries: Dict[
            utils.Stream, List[Tuple[utils.Link, int, Occupancy]]
        ] = {}
        self.link_occupancy: Dict[utils.Link, List[Occupancy]] = {
            link: [] for link in self.net.links
        }
        self.link_queue_occupancy: Dict[utils.Link, Dict[int, List[Occupancy]]] = {
            link: {queue: [] for queue in range(link.q_num)} for link in self.net.links
        }
        self.link_to_streams: Dict[utils.Link, Set[utils.Stream]] = {
            link: set() for link in self.net.links
        }

    def configure_attempt_state(self, attempt_index: int) -> None:
        self.local_search_nodes = 0
        self.search_budget_exhausted = False
        self.time_check_counter = 0
        self.active_wait_relax = min(attempt_index, 2)
        self.active_repair_wait_relax = 2 + attempt_index
        self.active_offset_scan_points = self.offset_scan_points + 4 * attempt_index
        self.active_max_offset_trials = self.max_offset_trials + 8 * attempt_index
        self.active_local_branch_limit = self.local_branch_limit + min(attempt_index, 2)
        self.active_local_search_limit = self.local_search_limit * (attempt_index + 1)

    @utils.check_time_limit
    def solve(self) -> utils.Statistics:
        if not self.is_prepared:
            self.prepare()

        start_time = utils.time_log()

        for stream in self.task.streams:
            if self.min_path_delay[stream] > stream.deadline:
                return utils.Statistics(
                    "-", utils.Result.unschedulable, utils.time_log() - start_time
                )

        heuristic_failed = False
        for attempt_index, stream_order in enumerate(
            self.stream_orders[: self.max_schedule_attempts]
        ):
            if utils.time_log() - start_time > utils.T_LIMIT:
                return utils.Statistics(
                    "-", utils.Result.unknown, utils.time_log() - start_time
                )
            self.start_attempt(attempt_index)
            if self.schedule_with_order(stream_order, start_time):
                return utils.Statistics(
                    "-", utils.Result.schedulable, utils.time_log() - start_time
                )
            heuristic_failed = True

        result = (
            utils.Result.unknown
            if heuristic_failed or utils.time_log() - start_time > utils.T_LIMIT
            else utils.Result.unschedulable
        )
        return utils.Statistics("-", result, utils.time_log() - start_time)

    def build_stream_orders(self) -> List[List[utils.Stream]]:
        def fail_first_key(stream: utils.Stream) -> Tuple[int, int, int, int, int]:
            slack = max(0, stream.deadline - self.min_path_delay[stream])
            hop_count = max(len(path.links) for path in self.candidate_paths[stream])
            return (
                len(self.candidate_paths[stream]),
                slack,
                stream.deadline,
                -hop_count,
                -stream.size,
            )

        orders = [
            list(self.base_stream_order),
            sorted(self.task.streams, key=fail_first_key),
            sorted(
                self.task.streams,
                key=lambda stream: (
                    stream.deadline,
                    -stream.size,
                    -self.stream_priority[stream],
                ),
            ),
            list(reversed(self.base_stream_order)),
        ]

        unique_orders: List[List[utils.Stream]] = []
        seen: Set[Tuple[utils.Stream, ...]] = set()
        for order in orders:
            key = tuple(order)
            if key in seen:
                continue
            seen.add(key)
            unique_orders.append(order)
        return unique_orders

    def start_attempt(self, attempt_index: int) -> None:
        self.reset_runtime_state()
        self.configure_attempt_state(attempt_index)

    def is_time_exhausted(self, start_time: float, force: bool = False) -> bool:
        if not force:
            self.time_check_counter += 1
            if self.time_check_counter % self.time_check_interval != 0:
                return False
        return utils.time_log() - start_time > utils.T_LIMIT

    def schedule_with_order(
        self, stream_order: List[utils.Stream], start_time: float
    ) -> bool:
        for stream in stream_order:
            if self.is_time_exhausted(start_time, force=True):
                self.search_budget_exhausted = True
                return False
            assignment = self.find_best_assignment(
                stream, relaxed_wait=self.active_wait_relax
            )
            if assignment is not None:
                self.commit_assignment(stream, assignment)
                continue
            if not self.repair_and_place(
                stream, start_time, relaxed_wait=self.active_repair_wait_relax
            ):
                return False
        return True

    def output(self) -> utils.Config:
        config = utils.Config()
        config.gcl = self.get_gcl()
        config.release = self.get_offset()
        config.route = self.get_route()
        config.queue = self.get_queue()
        config._delay = self.get_delay()
        return config

    def rank_paths(self, stream: utils.Stream) -> List[utils.Path]:
        scored_paths: List[Tuple[float, utils.Path]] = []
        for path in self.task_routes[stream]:
            path_key = tuple(path.links)
            delay = self.get_nw_delay_for_path(stream, path)
            hop_count = len(path.links)
            static_score = float(delay) + 0.5 * hop_count
            self.path_delay[(stream, path_key)] = delay
            self.path_static_score[(stream, path_key)] = static_score
            scored_paths.append((static_score, path))
        scored_paths.sort(key=lambda item: item[0])
        return [path for _, path in scored_paths]

    def get_stream_priority(self, stream: utils.Stream) -> float:
        slack = max(0, stream.deadline - self.min_path_delay.get(stream, 0))
        candidate_count = max(1, len(self.candidate_paths[stream]))
        max_hops = max(len(path.links) for path in self.candidate_paths[stream])
        return (
            10.0 / (slack + 1)
            + 4.0 / candidate_count
            + 0.05 * stream.size
            + 0.5 * max_hops
        )

    def get_min_path_delay(self, stream: utils.Stream) -> int:
        return self.min_path_delay[stream]

    def get_nw_delay_for_path(self, stream: utils.Stream, path: utils.Path) -> int:
        path_key = tuple(path.links)
        cached = self.path_delay.get((stream, path_key))
        if cached is not None:
            return cached
        delay = sum(link.t_proc + stream.get_t_trans(link) for link in path.links)
        self.path_delay[(stream, path_key)] = delay
        return delay

    def find_best_assignment(
        self, stream: utils.Stream, relaxed_wait: int = 0, branch_limit: Optional[int] = None
    ) -> Optional[Assignment]:
        candidate_limit = 1 if branch_limit is None else branch_limit
        candidates = self.enumerate_assignments(stream, relaxed_wait, candidate_limit)
        if not candidates:
            return None
        return candidates[0]

    def enumerate_assignments(
        self, stream: utils.Stream, relaxed_wait: int = 0, max_results: Optional[int] = None
    ) -> List[Assignment]:
        candidates: List[Assignment] = []
        for path in self.candidate_paths[stream]:
            delay = self.get_nw_delay_for_path(stream, path)
            if delay > stream.deadline:
                continue
            wait_budget = self.get_wait_budget(stream, path, relaxed_wait)
            for offset in self.get_candidate_offsets(stream, path):
                assignment = self.build_assignment(stream, path, offset, wait_budget)
                if assignment is not None:
                    self.consider_candidate(candidates, assignment, max_results)
        candidates.sort(key=lambda assignment: assignment.score)
        return candidates

    @staticmethod
    def consider_candidate(
        candidates: List[Assignment], assignment: Assignment, max_results: Optional[int]
    ) -> None:
        candidates.append(assignment)
        candidates.sort(key=lambda item: item.score)
        if max_results is not None and len(candidates) > max_results:
            del candidates[max_results:]

    def get_wait_budget(
        self, stream: utils.Stream, path: utils.Path, relaxed_wait: int
    ) -> int:
        path_delay = self.get_nw_delay_for_path(stream, path)
        slack = max(0, stream.deadline - path_delay)
        per_hop_slack = slack // max(1, len(path.links))
        return min(self.max_wait + relaxed_wait, per_hop_slack if per_hop_slack > 0 else relaxed_wait)

    def get_candidate_offsets(self, stream: utils.Stream, path: utils.Path) -> List[int]:
        period = stream.period
        first_link = path.links[0]
        frames = self.stream_frames[stream]
        offsets: Set[int] = {0}
        for _, end, _ in self.link_occupancy[first_link]:
            for frame in frames:
                candidate = end - frame * period
                if 0 <= candidate < period:
                    offsets.add(candidate)

        if len(offsets) < self.active_offset_scan_points:
            step = max(1, period // self.active_offset_scan_points)
            for candidate in range(0, period, step):
                offsets.add(candidate)
                if len(offsets) >= self.active_max_offset_trials:
                    break

        return sorted(offsets)[: self.active_max_offset_trials]

    def build_assignment(
        self, stream: utils.Stream, path: utils.Path, offset: int, wait_budget: int
    ) -> Optional[Assignment]:
        queue_map: Dict[utils.Link, int] = {}
        hop_starts: Dict[utils.Link, int] = {}
        hop_ends: Dict[utils.Link, int] = {}

        for index, link in enumerate(path.links):
            transmit = stream.get_t_trans(link)
            if index == 0:
                chosen_start = offset
                queue = self.find_queue_for_start(stream, link, chosen_start)
                if queue is None:
                    return None
            else:
                prev_link = path.links[index - 1]
                release = hop_ends[prev_link] + prev_link.t_proc
                chosen_start = -1
                queue = None
                for start in range(release, release + wait_budget + 1):
                    queue = self.find_queue_for_start(stream, link, start)
                    if queue is not None:
                        chosen_start = start
                        break
                if queue is None:
                    return None

            hop_starts[link] = chosen_start
            hop_ends[link] = chosen_start + transmit
            queue_map[link] = queue

        delay = hop_ends[path.links[-1]] - hop_starts[path.links[0]]
        if delay > stream.deadline:
            return None

        score = self.score_assignment(stream, path, queue_map, hop_starts, hop_ends, delay)
        return Assignment(path, offset, queue_map, hop_starts, hop_ends, delay, score)

    def find_queue_for_start(
        self, stream: utils.Stream, link: utils.Link, start: int
    ) -> Optional[int]:
        end = start + stream.get_t_trans(link)
        frames = self.stream_frames[stream]
        if not self.is_periodic_slot_free(self.link_occupancy[link], start, end, stream.period, frames):
            return None

        best_queue: Optional[int] = None
        best_cost: Optional[Tuple[int, int]] = None
        for queue, occupancy in self.link_queue_occupancy[link].items():
            if not self.is_periodic_slot_free(occupancy, start, end, stream.period, frames):
                continue
            cost = (len(occupancy), queue)
            if best_cost is None or cost < best_cost:
                best_cost = cost
                best_queue = queue
        return best_queue

    def is_periodic_slot_free(
        self,
        occupancy: List[Occupancy],
        start: int,
        end: int,
        period: int,
        frames: List[int],
    ) -> bool:
        if not occupancy:
            return True
        for frame in frames:
            abs_start = start + frame * period
            abs_end = end + frame * period
            for occ_start, occ_end, _ in occupancy:
                if occ_start >= abs_end:
                    break
                if abs_start < occ_end and abs_end > occ_start:
                    return False
        return True

    def score_assignment(
        self,
        stream: utils.Stream,
        path: utils.Path,
        queue_map: Dict[utils.Link, int],
        hop_starts: Dict[utils.Link, int],
        hop_ends: Dict[utils.Link, int],
        delay: int,
    ) -> float:
        queue_load = sum(len(self.link_queue_occupancy[link][queue_map[link]]) for link in path.links)
        waits = 0
        for index in range(1, len(path.links)):
            prev_link = path.links[index - 1]
            link = path.links[index]
            release = hop_ends[prev_link] + prev_link.t_proc
            waits += hop_starts[link] - release
        return float(delay) + 0.1 * queue_load + 0.2 * waits + 0.05 * len(path.links)

    def commit_assignment(self, stream: utils.Stream, assignment: Assignment) -> None:
        if stream in self.assignments:
            self.remove_assignment(stream)
        self.apply_assignment(stream, assignment)

    def apply_assignment(self, stream: utils.Stream, assignment: Assignment) -> None:
        self.assignments[stream] = assignment
        entries: List[Tuple[utils.Link, int, Occupancy]] = []
        touched_links: Set[utils.Link] = set()
        touched_queues: Dict[utils.Link, Set[int]] = {}

        for link in assignment.path.links:
            touched_links.add(link)
            self.link_to_streams[link].add(stream)
            queue = assignment.queue_map[link]
            touched_queues.setdefault(link, set()).add(queue)
            start = assignment.hop_starts[link]
            end = assignment.hop_ends[link]
            for frame in self.stream_frames[stream]:
                entry = (start + frame * stream.period, end + frame * stream.period, stream)
                self.link_occupancy[link].append(entry)
                self.link_queue_occupancy[link][queue].append(entry)
                entries.append((link, queue, entry))

        self.assignment_entries[stream] = entries
        for link in touched_links:
            self.link_occupancy[link].sort(key=lambda item: item[0])
        for link, queues in touched_queues.items():
            for queue in queues:
                self.link_queue_occupancy[link][queue].sort(key=lambda item: item[0])

    def remove_assignment(self, stream: utils.Stream) -> None:
        assignment = self.assignments.pop(stream, None)
        entries = self.assignment_entries.pop(stream, None)
        if assignment is None or entries is None:
            return

        touched_links: Set[utils.Link] = set()
        for link, queue, entry in entries:
            touched_links.add(link)
            self.link_occupancy[link].remove(entry)
            self.link_queue_occupancy[link][queue].remove(entry)

        for link in touched_links:
            self.link_to_streams[link].discard(stream)

    def rebuild_resources(self) -> None:
        assignments = list(self.assignments.items())
        self.reset_runtime_state()

        for stream, assignment in assignments:
            self.apply_assignment(stream, assignment)

    def repair_and_place(
        self, stream: utils.Stream, start_time: float, relaxed_wait: int
    ) -> bool:
        neighborhood = self.collect_repair_neighborhood(stream)
        if not neighborhood:
            return False

        removed_assignments = {
            neighbor: self.assignments[neighbor]
            for neighbor in neighborhood
            if neighbor in self.assignments
        }
        for neighbor in neighborhood:
            self.remove_assignment(neighbor)

        candidates = sorted(
            neighborhood | {stream},
            key=lambda item: self.stream_priority[item],
            reverse=True,
        )
        if self.solve_local_reschedule_gurobi(candidates, start_time, relaxed_wait):
            return True
        if self.local_reschedule(candidates, start_time, relaxed_wait=relaxed_wait):
            return True

        for neighbor, assignment in removed_assignments.items():
            self.apply_assignment(neighbor, assignment)
        return False

    def collect_repair_neighborhood(self, stream: utils.Stream) -> Set[utils.Stream]:
        counts: Dict[utils.Stream, int] = {}
        for path in self.candidate_paths[stream][: self.repair_path_budget]:
            for link in path.links:
                for neighbor in self.link_to_streams[link]:
                    counts[neighbor] = counts.get(neighbor, 0) + 1

        ranked = sorted(
            counts.items(),
            key=lambda item: (item[1], self.stream_priority[item[0]]),
            reverse=True,
        )
        return {neighbor for neighbor, _ in ranked[: self.repair_neighborhood]}

    def solve_local_reschedule_gurobi(
        self, streams: List[utils.Stream], start_time: float, relaxed_wait: int
    ) -> bool:
        if self.is_time_exhausted(start_time):
            self.search_budget_exhausted = True
            return False

        path_offset_options: Dict[utils.Stream, Dict[int, List[LocalRepairOption]]] = {}
        per_path_limit = max(
            self.active_local_branch_limit, self.gurobi_repair_candidate_limit
        )
        for stream in streams:
            options = self.collect_local_repair_options(
                stream, relaxed_wait, per_path_limit
            )
            if not options:
                return False
            path_offset_options[stream] = options

        try:
            model = gp.Model("jrs_bwq_local_repair")
            model.Params.LogToConsole = 0
            model.Params.Threads = self.workers
            remaining_time = max(0.05, utils.T_LIMIT - utils.time_log())
            model.Params.TimeLimit = min(self.gurobi_repair_time_limit, remaining_time)

            path_vars: Dict[Tuple[utils.Stream, int], gp.Var] = {}
            offset_vars: Dict[Tuple[utils.Stream, int, int], gp.Var] = {}

            for stream, options_by_path in path_offset_options.items():
                for path_index, options in options_by_path.items():
                    path_vars[(stream, path_index)] = model.addVar(
                        vtype=gp.GRB.BINARY,
                        name=f"path_{int(stream)}_{path_index}",
                    )
                    for option_index, _ in enumerate(options):
                        offset_vars[(stream, path_index, option_index)] = model.addVar(
                            vtype=gp.GRB.BINARY,
                            name=f"offset_{int(stream)}_{path_index}_{option_index}",
                        )

            model.update()

            for stream, options_by_path in path_offset_options.items():
                model.addConstr(
                    gp.quicksum(
                        path_vars[(stream, path_index)]
                        for path_index in options_by_path
                    )
                    == 1,
                    name=f"path_select_{int(stream)}",
                )
                for path_index, options in options_by_path.items():
                    model.addConstr(
                        gp.quicksum(
                            offset_vars[(stream, path_index, option_index)]
                            for option_index in range(len(options))
                        )
                        == path_vars[(stream, path_index)],
                        name=f"offset_select_{int(stream)}_{path_index}",
                    )

            stream_list = list(streams)
            for left_index, left_stream in enumerate(stream_list):
                left_options_by_path = path_offset_options[left_stream]
                for right_stream in stream_list[left_index + 1 :]:
                    right_options_by_path = path_offset_options[right_stream]
                    for left_path_index, left_options in left_options_by_path.items():
                        for right_path_index, right_options in right_options_by_path.items():
                            for left_option_index, left_option in enumerate(left_options):
                                for right_option_index, right_option in enumerate(right_options):
                                    if self.assignments_conflict(
                                        left_stream,
                                        left_option.assignment,
                                        right_stream,
                                        right_option.assignment,
                                    ):
                                        model.addConstr(
                                            offset_vars[(left_stream, left_path_index, left_option_index)]
                                            + offset_vars[(right_stream, right_path_index, right_option_index)]
                                            <= 1,
                                            name=(
                                                f"conflict_{int(left_stream)}_{left_path_index}_{left_option_index}_"
                                                f"{int(right_stream)}_{right_path_index}_{right_option_index}"
                                            ),
                                        )

            model.setObjective(
                gp.quicksum(
                    option.assignment.score
                    * offset_vars[(stream, path_index, option_index)]
                    for stream, options_by_path in path_offset_options.items()
                    for path_index, options in options_by_path.items()
                    for option_index, option in enumerate(options)
                ),
                gp.GRB.MINIMIZE,
            )
            model.optimize()

            if model.Status not in {gp.GRB.OPTIMAL, gp.GRB.TIME_LIMIT}:
                return False
            if model.SolCount == 0:
                return False

            for stream, options_by_path in path_offset_options.items():
                selected_option: Optional[LocalRepairOption] = None
                for path_index, options in options_by_path.items():
                    for option_index, option in enumerate(options):
                        if offset_vars[(stream, path_index, option_index)].X > 0.5:
                            selected_option = option
                            break
                    if selected_option is not None:
                        break
                if selected_option is None:
                    return False
                self.apply_assignment(stream, selected_option.assignment)
            return True
        except gp.GurobiError:
            return False

    def collect_local_repair_options(
        self, stream: utils.Stream, relaxed_wait: int, per_path_limit: int
    ) -> Dict[int, List[LocalRepairOption]]:
        options_by_path: Dict[int, List[LocalRepairOption]] = {}
        for path_index, path in enumerate(self.candidate_paths[stream]):
            delay = self.get_nw_delay_for_path(stream, path)
            if delay > stream.deadline:
                continue
            wait_budget = self.get_wait_budget(stream, path, relaxed_wait)
            path_options: List[LocalRepairOption] = []
            for offset in self.get_candidate_offsets(stream, path):
                assignment = self.build_assignment(stream, path, offset, wait_budget)
                if assignment is None:
                    continue
                path_options.append(LocalRepairOption(path_index, offset, assignment))
                if len(path_options) >= per_path_limit:
                    break
            if path_options:
                options_by_path[path_index] = path_options
        return options_by_path

    def assignments_conflict(
        self,
        left_stream: utils.Stream,
        left_assignment: Assignment,
        right_stream: utils.Stream,
        right_assignment: Assignment,
    ) -> bool:
        shared_links = set(left_assignment.path.links) & set(right_assignment.path.links)
        if not shared_links:
            return False

        for link in shared_links:
            if self.periodic_intervals_conflict(
                left_assignment.hop_starts[link],
                left_assignment.hop_ends[link],
                left_stream.period,
                right_assignment.hop_starts[link],
                right_assignment.hop_ends[link],
                right_stream.period,
            ):
                return True
        return False

    @staticmethod
    def periodic_intervals_conflict(
        left_start: int,
        left_end: int,
        left_period: int,
        right_start: int,
        right_end: int,
        right_period: int,
    ) -> bool:
        left_length = left_end - left_start
        right_length = right_end - right_start
        phase_gcd = math.gcd(left_period, right_period)
        phase_delta = right_start - left_start

        lower_bound = -phase_delta - right_length + 1
        upper_bound = -phase_delta + left_length - 1
        min_multiple = math.ceil(lower_bound / phase_gcd)
        max_multiple = math.floor(upper_bound / phase_gcd)
        return min_multiple <= max_multiple

    def local_reschedule(
        self, pending: List[utils.Stream], start_time: float, relaxed_wait: int
    ) -> bool:
        if not pending:
            return True
        if self.is_time_exhausted(start_time):
            self.search_budget_exhausted = True
            return False
        self.local_search_nodes += 1
        if self.local_search_nodes > self.active_local_search_limit:
            self.search_budget_exhausted = True
            return False

        stream = pending[0]
        candidates = self.enumerate_assignments(
            stream, relaxed_wait, self.active_local_branch_limit
        )
        for assignment in candidates:
            self.apply_assignment(stream, assignment)
            if self.local_reschedule(pending[1:], start_time, relaxed_wait):
                return True
            self.remove_assignment(stream)
        return False

    def get_gcl(self) -> utils.GCL:
        gcl = []
        for stream, assignment in self.assignments.items():
            for link in assignment.path.links:
                queue = assignment.queue_map[link]
                start = assignment.hop_starts[link]
                end = assignment.hop_ends[link]
                for frame in self.stream_frames[stream]:
                    gcl.append(
                        [
                            link,
                            queue,
                            start + frame * stream.period,
                            end + frame * stream.period,
                            self.task.lcm,
                        ]
                    )
        return utils.GCL(gcl)

    def get_offset(self) -> utils.Release:
        offset = []
        for stream, assignment in self.assignments.items():
            offset.append([stream, 0, assignment.offset])
        return utils.Release(offset)

    def get_route(self) -> utils.Route:
        route = []
        for stream, assignment in self.assignments.items():
            for link in assignment.path.links:
                route.append([stream, link])
        return utils.Route(route)

    def get_queue(self) -> utils.Queue:
        queue = []
        for stream, assignment in self.assignments.items():
            for link in assignment.path.links:
                queue.append([stream, 0, link, assignment.queue_map[link]])
        return utils.Queue(queue)

    def get_delay(self) -> utils.Delay:
        delay = []
        for stream, assignment in self.assignments.items():
            delay.append([stream, 0, assignment.delay])
        return utils.Delay(delay)


if __name__ == "__main__":
    args = utils.parse_command_line_args()
    utils.Statistics().header()
    benchmark(args.name, args.task, args.net, args.output, args.workers)