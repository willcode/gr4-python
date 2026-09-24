"""Tests that the type stub states every name the extension exports, with a docstring."""

import ast
import enum
import os
import sys
import unittest

from gr4 import _gr4

STUB = os.path.join(os.path.dirname(_gr4.__file__), "_gr4.pyi")


def public(names):
    """Returns the names that do not start with an underscore."""
    return {name for name in names if not name.startswith("_")}


def member_names(node):
    """Returns the names a class body of the stub defines, with the value of each constant assignment."""
    members = {}
    for statement in node.body:
        if isinstance(statement, ast.FunctionDef):
            members[statement.name] = None
        elif isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
            members[statement.target.id] = None
        elif isinstance(statement, ast.Assign):
            for target in statement.targets:
                members[target.id] = statement.value.value if isinstance(statement.value, ast.Constant) else None
    return members


class Stub(unittest.TestCase):
    """The stub and the extension export the same names."""

    @classmethod
    def setUpClass(cls):
        with open(STUB, encoding="utf-8") as stub:
            cls.tree = ast.parse(stub.read())
        cls.classes = {node.name: node for node in cls.tree.body if isinstance(node, ast.ClassDef)}
        cls.functions = {node.name: node for node in cls.tree.body if isinstance(node, ast.FunctionDef)}
        cls.constants = {
            node.target.id
            for node in cls.tree.body
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
        }

    def test_the_stub_names_every_export_and_nothing_else(self):
        stated = public(self.classes) | public(self.functions) | public(self.constants)
        self.assertEqual(stated, public(dir(_gr4)))

    def test_each_class_states_the_members_of_the_extension_class(self):
        for name, node in self.classes.items():
            with self.subTest(name=name):
                stated = member_names(node)
                bound = getattr(_gr4, name)
                if issubclass(bound, enum.Enum):
                    self.assertEqual(
                        {member: stated[member] for member in public(stated)}, {m.name: m.value for m in bound}
                    )
                elif issubclass(bound, BaseException):
                    self.assertEqual(public(stated), public(vars(bound("message"))))
                else:
                    self.assertEqual(public(stated), public(vars(bound)))

    def test_the_module_and_every_class_and_function_carry_a_docstring(self):
        self.assertTrue(ast.get_docstring(self.tree))
        for node in ast.walk(self.tree):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                with self.subTest(name=node.name):
                    self.assertTrue(ast.get_docstring(node))


if __name__ == "__main__":
    # os._exit skips static destruction in the core and plugin libraries, whose order across libraries is unspecified
    program = unittest.main(exit=False, verbosity=2)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0 if program.result.wasSuccessful() else 1)
