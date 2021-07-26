from timeit import default_timer as timer


class StopwatchPrint(object):
    def __init__(self,
                 start_print="Start stopwatch",
                 final_print="Elapsed time %2.4f"):
        self.start_print = start_print
        self.final_print = final_print

    def __enter__(self):
        self.tic = timer()
        print(self.start_print)
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.toc = timer()
        print(self.final_print % (self.toc-self.tic))