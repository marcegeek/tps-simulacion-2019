import random

from code.simulation import Event, Simulation
from code.stats.distributions import ExponentialDistribution


class MMc(Simulation):
    STATUS_IDLE, STATUS_BUSY = 0, 1

    def __init__(self, servers, arrival_rate, service_rate, seed=None):
        super().__init__()
        self.arrival_rate = arrival_rate
        self.service_rate = service_rate
        self.arr_dist = ExponentialDistribution(self.arrival_rate, seed=seed)
        self.svc_dist = ExponentialDistribution(self.service_rate, seed=seed)
        self.schedule_next_arrival()

        self.arrival_times = []
        self.statuses = [self.STATUS_IDLE] * servers
        self.queue_len = 0

        self.customers_delayed = 0
        self.total_delay = 0.
        self.num_in_queue_area = 0.
        self.statuses_area = 0.

    def schedule_next_arrival(self):
        self.events.append(
            Event(self.clock + self.arr_dist.random_sample(), self.arrival)
        )

    def schedule_next_departure(self, server):
        ev = Event(self.clock + self.svc_dist.random_sample(), self.departure)
        ev.server = server
        self.events.append(ev)

    def arrival(self, ev):
        self.schedule_next_arrival()
        if self.statuses.count(self.STATUS_BUSY) == len(self.statuses):
            self.queue_len += 1
            self.arrival_times.append(ev.time)
        else:
            self.customers_delayed += 1
            idle_servers = [i for i in range(len(self.statuses)) if self.statuses[i] == self.STATUS_IDLE]
            index = random.choice(idle_servers)
            self.statuses[index] = self.STATUS_BUSY
            self.schedule_next_departure(index)

    def departure(self, ev):
        self.statuses[ev.server] = self.STATUS_IDLE
        if self.queue_len != 0:
            self.queue_len -= 1
            self.total_delay += ev.time - self.arrival_times.pop(0)
            self.customers_delayed += 1
            idle_servers = [i for i in range(len(self.statuses)) if self.statuses[i] == self.STATUS_IDLE]
            index = random.choice(idle_servers)
            self.statuses[index] = self.STATUS_BUSY
            self.schedule_next_departure(index)

    def update_stats(self):
        super().update_stats()
        self.num_in_queue_area += self.queue_len * self.time_since_last_event
        self.statuses_area += sum(self.statuses)/len(self.statuses) * self.time_since_last_event

    def average_delay(self):
        return self.total_delay / self.customers_delayed

    def average_number_in_queue(self):
        return self.num_in_queue_area / self.clock

    def server_utilization(self):
        return self.statuses_area / self.clock

    def run(self, delays_required):
        while self.customers_delayed < delays_required:
            self.run_once()

    def report(self):
        print('Average delay in queue{:11.3f} minutes'.format(self.average_delay()))
        print('Average number in queue{:10.3f}'.format(self.average_number_in_queue()))
        print('Server utilization{:15.3f}'.format(self.server_utilization()))
        print('Time simulation ended{:12.3f}'.format(self.clock))


class MM1(MMc):

    def __init__(self, arrival_rate, service_rate, seed=None):
        super().__init__(1, arrival_rate, service_rate, seed=seed)


def main():
    sim = MM1(2, 4)
    sim.run(1000)
    print('M/M/1:')
    sim.report()
    print()

    sim = MMc(2, 2, 4)
    sim.run(1000)
    print('M/M/2:')
    sim.report()


if __name__ == '__main__':
    main()
