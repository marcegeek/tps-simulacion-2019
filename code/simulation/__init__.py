import abc


class Event(abc.ABC):

    def __init__(self, time, method):
        self.time = time
        self.method = method

    def __lt__(self, other):
        return self.time < other.time

    def __gt__(self, other):
        return self.time > other.time

    def handle(self):
        self.method(self)


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
        self.next_event.handle()
        self.events.remove(self.next_event)
        self.next_event = None
