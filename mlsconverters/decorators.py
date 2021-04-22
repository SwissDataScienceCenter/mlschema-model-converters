import inspect


class params:
    @staticmethod
    def _get_locals_up_one_frame():
        return inspect.getouterframes(inspect.currentframe())[2][0].f_locals

    def __enter__(self):
        outer_locals = self._get_locals_up_one_frame()
        self.l1 = set(outer_locals.keys())

    def __exit__(self, exc_type, exc_value, exc_traceback):
        outer_locals = self._get_locals_up_one_frame()
        self.l2 = set(outer_locals.keys())
        self.params = {k: outer_locals[k] for k in self.l2.difference(self.l1)}
        print(self.params)
