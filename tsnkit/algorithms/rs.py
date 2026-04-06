
import traceback
from typing import Dict, List, Optional, Set, Tuple

from .. import core as utils


StateSnapshot = Tuple[
    Dict[utils.Stream, utils.Path],
    Dict[utils.Stream, int],
    Dict[utils.Stream, int],
    Dict[Tuple[utils.Stream, utils.Link], int],
    Dict[utils.Link, List[List[int]]],
    Dict[utils.Link, Dict[int, List[List[int]]]],
]


def benchmark(name, task_path, net_path, output_path="./", workers=1) -> utils.Statistics:
    stat = utils.Statistics(name)
    try:
        test = rs(workers)
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
        print("[!", e, "]", flush=True)
        traceback.print_exc()
        stat.result = utils.Result.error
        stat.content(name=name)
        return stat


class rs:
    def __init__(self, workers=1) -> None:
        self.workers = workers
        self.max_branch_per_stream = 3
        self.max_search_nodes = 2000

    def init(self, task_path: str, net_path: str) -> None:
        self.task = utils.load_stream(task_path)
        self.net = utils.load_network(net_path)

        self.task_routes = {
            s: self.net.get_all_path(s.src, s.dst) for s in self.task
        }

        ## Initialize routing map, will be set during solve
        self.task_routing_map: Dict[utils.Stream, utils.Path] = {}

        self.offset_map: Dict[utils.Stream, int] = {}
        self.network_delay: Dict[utils.Stream, int] = {}
        self.queue_map: Dict[Tuple[utils.Stream, utils.Link], int] = {}

        ## Per-link used range list: each entry is (start, end)
        self.link_occupancy: Dict[utils.Link, List[List[int]]] = {
            l: [] for l in self.net.links
        }
        self.link_queue_occupancy: Dict[utils.Link, Dict[int, List[List[int]]]] = {
            l: {q: [] for q in range(l.q_num)} for l in self.net.links
        }

    def prepare(self) -> None:
        pass

    @utils.check_time_limit
    def solve(self) -> utils.Statistics:
        start_time = utils.time_log()

        self.search_nodes = 0
        self.search_budget_exhausted = False
        self.max_search_nodes = max(2000, len(self.task.streams) * 400)

        for s in self.task:
            if self.get_min_path_delay(s) > s.deadline:
                return utils.Statistics(
                    "-", utils.Result.unschedulable, utils.time_log() - start_time
                )

        if self.search_schedule(set(self.task.streams), start_time):
            return utils.Statistics(
                "-", utils.Result.schedulable, utils.time_log() - start_time
            )

        result = (
            utils.Result.unknown
            if self.search_budget_exhausted or utils.time_log() - start_time > utils.T_LIMIT
            else utils.Result.unschedulable
        )
        return utils.Statistics("-", result, utils.time_log() - start_time)

    def search_schedule(self, unscheduled: Set[utils.Stream], start_time: float) -> bool:
        if not unscheduled:
            return True

        if utils.time_log() - start_time > utils.T_LIMIT:
            self.search_budget_exhausted = True
            return False

        self.search_nodes += 1
        if self.search_nodes > self.max_search_nodes:
            self.search_budget_exhausted = True
            return False

        candidate_info = []
        for s in unscheduled:
            candidates = self.get_schedule_candidates(s)
            if not candidates:
                return False
            ordered_candidates = sorted(candidates, key=lambda x: (x[3], x[2], x[1]))
            candidate_info.append(
                (len(ordered_candidates), min(c[3] for c in ordered_candidates), s, ordered_candidates)
            )

        candidate_info.sort(
            key=lambda x: (x[0], x[1], x[2].deadline, x[2].period, x[2].size)
        )
        _, _, s, candidates = candidate_info[0]

        for candidate in candidates[: self.max_branch_per_stream]:
            snapshot = self.snapshot_state()
            path, offset, delay, _, queue_map = candidate
            self.assign_stream(s, path, offset, delay, queue_map)
            next_unscheduled = set(unscheduled)
            next_unscheduled.remove(s)
            if self.search_schedule(next_unscheduled, start_time):
                return True
            self.restore_state(snapshot)

        return False

    def assign_stream(
        self,
        s: utils.Stream,
        path: utils.Path,
        offset: int,
        delay: int,
        queue_map: Dict[utils.Link, int],
    ) -> None:
        self.task_routing_map[s] = path
        self.network_delay[s] = delay
        self.offset_map[s] = offset
        for link, queue in queue_map.items():
            self.queue_map[(s, link)] = queue
        self.commit_stream(s, offset)

    def snapshot_state(self) -> StateSnapshot:
        return (
            dict(self.task_routing_map),
            dict(self.offset_map),
            dict(self.network_delay),
            dict(self.queue_map),
            {
                link: [interval[:] for interval in intervals]
                for link, intervals in self.link_occupancy.items()
            },
            {
                link: {
                    queue: [interval[:] for interval in intervals]
                    for queue, intervals in queue_dict.items()
                }
                for link, queue_dict in self.link_queue_occupancy.items()
            },
        )

    def restore_state(self, snapshot: StateSnapshot) -> None:
        (
            task_routing_map,
            offset_map,
            network_delay,
            queue_map,
            link_occupancy,
            link_queue_occupancy,
        ) = snapshot

        self.task_routing_map = dict(task_routing_map)
        self.offset_map = dict(offset_map)
        self.network_delay = dict(network_delay)
        self.queue_map = dict(queue_map)
        self.link_occupancy = {
            link: [interval[:] for interval in intervals]
            for link, intervals in link_occupancy.items()
        }
        self.link_queue_occupancy = {
            link: {
                queue: [interval[:] for interval in intervals]
                for queue, intervals in queue_dict.items()
            }
            for link, queue_dict in link_queue_occupancy.items()
        }

    def get_schedule_candidates(
        self, s: utils.Stream
    ) -> List[Tuple[utils.Path, int, int, float, Dict[utils.Link, int]]]:
        """Return feasible (path, offset, delay, cost, queue_map) tuples for stream s."""
        candidates = []
        for path in self.task_routes[s]:
            delay = self.get_nw_delay_for_path(s, path)
            if delay > s.deadline:
                continue

            path_load = self.get_path_load(path)
            offset, queue_assignment = self.find_inject_offset_for_path(s, path)
            if offset >= 0 and queue_assignment is not None:
                cost = delay + 0.01 * path_load
                candidates.append((path, offset, delay, cost, queue_assignment))

        return candidates

    def get_path_load(self, path: utils.Path) -> int:
        """Estimate current congestion on a path."""
        return sum(len(self.link_occupancy[l]) for l in path.links)

    def get_nw_delay_for_path(self, s: utils.Stream, path: utils.Path) -> int:
        """Calculate network delay for a specific path."""
        return sum(l.t_proc + s.get_t_trans(l) for l in path.links)

    def get_min_path_delay(self, s: utils.Stream) -> int:
        return min(self.get_nw_delay_for_path(s, path) for path in self.task_routes[s])

    def find_inject_offset_for_path(
        self, s: utils.Stream, path: utils.Path
    ) -> Tuple[int, Optional[Dict[utils.Link, int]]]:
        """Find feasible injection offset and queue assignment for a specific path."""
        max_offset = s.period
        for o in range(0, max_offset):
            feasible, queue_assignment = self.check_offset_feasible_for_path(
                s, path, o
            )
            if feasible:
                return o, queue_assignment
        return -1, None

    def check_offset_feasible_for_path(
        self, s: utils.Stream, path: utils.Path, o: int
    ) -> Tuple[bool, Optional[Dict[utils.Link, int]]]:
        """Check if offset o is feasible for stream s on path and assign queues if possible."""
        lcm = self.task.lcm
        frames = s.get_frame_indexes(lcm)
        intervals_by_link: Dict[utils.Link, List[List[int]]] = {
            l: [] for l in path.links
        }
        queue_assignment: Dict[utils.Link, int] = {}

        for k in frames:
            t_base = o + k * s.period
            prev_end = t_base
            for l in path.links:
                transmit = s.get_t_trans(l)
                start = prev_end
                end = start + transmit
                intervals_by_link[l].append([start, end])

                # Check against existing occupancy on this link
                for occupied in self.link_occupancy[l]:
                    if not (end <= occupied[0] or start >= occupied[1]):
                        return False, None

                prev_end = end + l.t_proc

        for l in path.links:
            queue_candidates = sorted(
                self.link_queue_occupancy[l].items(),
                key=lambda item: (0 if len(item[1]) == 0 else 1, len(item[1]), item[0]),
            )
            assigned_queue = None
            for q, occupied_list in queue_candidates:
                if all(
                    self.intervals_do_not_overlap(interval, occupied)
                    for interval in intervals_by_link[l]
                    for occupied in occupied_list
                ):
                    assigned_queue = q
                    break
            if assigned_queue is None:
                return False, None
            queue_assignment[l] = assigned_queue

        return True, queue_assignment

    def commit_stream(self, s: utils.Stream, o: int) -> None:
        """Commit the stream's schedule to link and queue occupancy."""
        path = self.task_routing_map[s]
        lcm = self.task.lcm
        frames = s.get_frame_indexes(lcm)

        for k in frames:
            t_base = o + k * s.period
            prev_end = t_base
            for l in path.links:
                transmit = s.get_t_trans(l)
                start = prev_end
                end = start + transmit
                self.link_occupancy[l].append([start, end])
                queue = self.queue_map[(s, l)]
                self.link_queue_occupancy[l][queue].append([start, end])
                prev_end = end + l.t_proc

        for l in path.links:
            self.link_occupancy[l].sort(key=lambda x: x[0])
            for q in self.link_queue_occupancy[l]:
                self.link_queue_occupancy[l][q].sort(key=lambda x: x[0])

    def output(self) -> utils.Config:
        config = utils.Config()
        config.gcl = self.get_gcl()
        config.release = self.get_offset()
        config.queue = self.get_queue()
        config.route = self.get_route()
        config._delay = self.get_delay()
        return config

    def get_gcl(self) -> utils.GCL:
        gcl = []
        lcm = self.task.lcm
        for s in self.task:
            if s not in self.offset_map:
                continue
            path = self.task_routing_map[s]
            off = self.offset_map[s]
            prev_end = off
            for l in path.links:
                start = prev_end
                end = start + s.get_t_trans(l)
                queue = self.queue_map.get((s, l), 0)
                for k in s.get_frame_indexes(lcm):
                    gcl.append([l, queue, start + k * s.period, end + k * s.period, lcm])
                prev_end = end + l.t_proc
        return utils.GCL(gcl)

    def get_offset(self) -> utils.Release:
        offset = []
        for s, off in self.offset_map.items():
            offset.append([s, 0, off])
        return utils.Release(offset)

    def get_queue(self) -> utils.Queue:
        queue = []
        for s in self.task:
            if s in self.task_routing_map:
                for l in self.task_routing_map[s].links:
                    queue.append([s, 0, l, self.queue_map.get((s, l), 0)])
        return utils.Queue(queue)

    def get_route(self) -> utils.Route:
        route = []
        for s in self.task:
            if s in self.task_routing_map:
                for l in self.task_routing_map[s].links:
                    route.append([s, l])
        return utils.Route(route)

    def get_delay(self) -> utils.Delay:
        delay = []
        for s in self.task:
            if s not in self.offset_map:
                continue
            path = self.task_routing_map[s]
            # End-to-end delay: total network hop delay excluding first link burst start
            delay_ns = self.network_delay[s] - path.links[0].t_proc - s.get_t_trans(path.links[0])
            delay.append([s, 0, delay_ns])
        return utils.Delay(delay)

    @staticmethod
    def intervals_do_not_overlap(a: List[int], b: List[int]) -> bool:
        return a[1] <= b[0] or a[0] >= b[1]


if __name__ == "__main__":
    args = utils.parse_command_line_args()
    utils.Statistics().header()
    benchmark(args.name, args.task, args.net, args.output, args.workers)
