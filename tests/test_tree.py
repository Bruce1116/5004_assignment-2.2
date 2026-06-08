import unittest

from numcompute_stream.tree import DecisionTreeClassifier


class TreeTests(unittest.TestCase):
    def test_tree_partial_fit(self):
        tree = DecisionTreeClassifier(max_depth=2)
        tree.partial_fit([[0], [1], [2]], [0, 0, 1])
        self.assertIsNotNone(tree.tree)

    def test_tree_predict_length(self):
        tree = DecisionTreeClassifier(max_depth=2)
        tree.partial_fit([[0], [1], [2]], [0, 0, 1])
        self.assertEqual(len(tree.predict([[0], [2]])), 2)

    def test_tree_supports_entropy(self):
        tree = DecisionTreeClassifier(max_depth=2, criterion="entropy")
        tree.partial_fit([[0], [1], [2]], [0, 0, 1])
        self.assertEqual(len(tree.predict([[1]])), 1)

    def test_tree_single_class(self):
        tree = DecisionTreeClassifier(max_depth=2)
        tree.partial_fit([[0], [1]], [1, 1])
        self.assertEqual(tree.predict([[2]])[0], 1)


if __name__ == "__main__":
    unittest.main()
