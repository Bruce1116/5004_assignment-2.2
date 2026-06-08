class Pipeline:
    def __init__(self, steps):
        self.steps = steps

    def partial_fit(self, X, y):
        data = X
        for name, step in self.steps[:-1]:
            if hasattr(step, "partial_fit"):
                step.partial_fit(data, y)
            if hasattr(step, "transform"):
                data = step.transform(data)

        model = self.steps[-1][1]
        model.partial_fit(data, y)
        return self

    def predict(self, X):
        data = X
        for name, step in self.steps[:-1]:
            data = step.transform(data)
        model = self.steps[-1][1]
        return model.predict(data)

    def fit_chunk(self, X, y):
        return self.partial_fit(X, y)

    def score_chunk(self, X, y):
        from .metrics import accuracy

        pred = self.predict(X)
        return accuracy(y, pred)

