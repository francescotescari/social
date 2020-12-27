import asyncio
import threading
import time
from concurrent.futures.thread import ThreadPoolExecutor
from queue import Queue, Empty


class ThreadSafeCounter:
    def __init__(self):
        self.lock = threading.Lock()
        self.value = 0

    def increment(self):
        with self.lock:
            self.value += 1

    def decrement(self):
        with self.lock:
            self.value -= 1


class AsyncMapper:

    def __init__(self, apply_function, input_generator, concurrent=1):
        self.queue = Queue()
        self.counter = ThreadSafeCounter()
        self.done = False
        self.executor = ThreadPoolExecutor(concurrent)
        self.async_gen = map(apply_function, input_generator)
        self.exception = None

        while self.counter.value < concurrent:
            self._start_next()
            self.counter.increment()

    def _start_next(self):
        try:
            coro = next(self.async_gen)
        except StopIteration:
            self.done = True
            return
        asyncio.get_event_loop().run_in_executor(self.executor, self._run_task, coro)

    def _run_task(self, task):
        try:
            result = asyncio.run(task)
        except Exception as e:
            self.exception = e
        else:
            self.queue.put(result)

    def __next__(self):
        if self.exception is not None:
            raise self.exception
        if self.done and self.counter.value == 0:
            raise StopIteration()
        result = self.queue.get()
        self.counter.decrement()
        self._start_next()
        return result

    def __iter__(self):
        return self


def async_map(apply_function, input_generator, executor):
    loop = asyncio.get_event_loop()
    async_gen = map(apply_function, input_generator)

    queue = Queue()
    counter = ThreadSafeCounter()

    done = {0: False}

    def _start_coroutines():
        def run_task(task):
            try:
                result = asyncio.run(task)
            except Exception as e:
                queue.put(e)
            else:
                queue.put(result)

        try:
            for coroutine in async_gen:
                counter.increment()
                loop.run_in_executor(executor, run_task, coroutine)
        except Exception as e:
            queue.put(e)
        else:
            done[0] = True

    loop.run_in_executor(executor, _start_coroutines)

    while not done[0] or counter.value > 0:
        data = queue.get()
        if isinstance(data, Exception):
            raise data
        yield data
        counter.decrement()
