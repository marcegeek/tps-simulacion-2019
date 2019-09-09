import abc

from code.stats.distributions import ExponentialDistribution


class Event(abc.ABC):

    def __init__(self, time, ev_type):
        self.time = time
        self.type = ev_type

    def __lt__(self, other):
        return self.time < other.time

    def __gt__(self, other):
        return self.time > other.time


class Simulation(abc.ABC):

    def __init__(self):
        self.clock = 0.
        self.events = []
        self.next_event = None  # el valor se modifica en la rutina de avance en el tiempo
        self.last_event_time = 0.
        self.time_since_last_event = 0.

    def timing(self):
        if len(self.events) == 0:
            print('Event list empty at time {}'.format(self.clock))
            raise Exception('Event list empty')
        self.next_event = min(self.events)
        self.clock = self.next_event.time

    def update_stats(self):
        self.time_since_last_event = self.clock - self.last_event_time
        self.last_event_time = self.clock

    def run_once(self):
        self.timing()
        self.update_stats()
        self.handle_event(self.next_event)

    @abc.abstractmethod
    def handle_event(self, event):
        pass


class MM1(Simulation):
    STATUS_IDLE, STATUS_BUSY = 0, 1
    ARRIVAL_EVENT, DEPARTURE_EVENT = 0, 1

    def __init__(self, mean_interarrival, mean_service, seed=None):
        super().__init__()
        self.mean_interarrival = mean_interarrival
        self.mean_service = mean_service
        self.arr_dist = ExponentialDistribution(1 / self.mean_interarrival, seed=seed)
        self.svc_dist = ExponentialDistribution(1 / self.mean_service, seed=seed)
        self.schedule_next_arrival()

        self.arrival_times = []
        self.status = self.STATUS_IDLE
        self.queue_len = 0

        self.customers_delayed = 0
        self.total_delay = 0.
        self.num_in_queue_area = 0.
        self.status_area = 0.

    def schedule_next_arrival(self):
        self.events.append(
            Event(self.clock + self.arr_dist.random_sample(), self.ARRIVAL_EVENT)
        )

    def schedule_next_departure(self):
        self.events.append(
            Event(self.clock + self.svc_dist.random_sample(),  self.DEPARTURE_EVENT)
        )

    def arrival(self):
        self.schedule_next_arrival()
        if self.status == self.STATUS_BUSY:
            self.queue_len += 1
            self.arrival_times.append(self.clock)
        else:
            self.customers_delayed += 1
            self.status = self.STATUS_BUSY
            self.schedule_next_departure()

    def departure(self):
        if self.queue_len == 0:
            self.status = self.STATUS_IDLE
        else:
            self.queue_len -= 1
            self.total_delay += self.clock - self.arrival_times.pop(0)
            self.customers_delayed += 1
            self.schedule_next_departure()

    def handle_event(self, event):
        if event.type == self.ARRIVAL_EVENT:
            self.arrival()
        elif event.type == self.DEPARTURE_EVENT:
            self.departure()
        self.events.remove(event)

    def update_stats(self):
        super().update_stats()
        self.num_in_queue_area += self.queue_len * self.time_since_last_event
        self.status_area += self.status * self.time_since_last_event

    def run(self, delays_required):
        while self.customers_delayed < delays_required:
            self.run_once()

    def report(self):
        print('Average delay in queue{:11.3f} minutes'.format(self.total_delay / self.customers_delayed))
        print('Average number in queue{:10.3f}'.format(self.num_in_queue_area / self.clock))
        print('Server utilization{:15.3f}'.format(self.status_area / self.clock))
        print('Time simulation ended{:12.3f}'.format(self.clock))

    def __eq__(self, other):
        return (self.status == other.status and
                self.queue_len == other.queue_len and
                self.last_event_time == other.last_event_time and
                self.customers_delayed == other.customers_delayed and
                self.total_delay == other.total_delay and
                self.num_in_queue_area == other.num_in_queue_area and
                self.status_area == other.status_area)


def main():
    mm1 = MM1(4, 3, seed=32145)
    mm1.run(1000)
    mm1.report()


if __name__ == '__main__':
    main()
