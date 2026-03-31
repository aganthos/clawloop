def pytest_collection_modifyitems(session, config, items):
    """Guard: public tests must not import private modules.

    Note: This AST-based check catches static imports only. Dynamic imports
    via importlib.import_module() or __import__() are not detected here.
    CI provides additional coverage for repository boundary checks.
    """
    import ast
    from pathlib import Path

    for item in items:
        fpath = Path(item.fspath)
        try:
            tree = ast.parse(fpath.read_text())
        except Exception:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and (
                node.module.startswith("private_")
            ):
                raise ValueError(
                    f"BOUNDARY VIOLATION: {fpath} imports from private code "
                    f"(line {node.lineno}). Public tests must not depend on "
                    f"non-public modules."
                )
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("private_"):
                        raise ValueError(
                            f"BOUNDARY VIOLATION: {fpath} imports from private code "
                            f"(line {node.lineno}). Public tests must not depend on "
                            f"non-public modules."
                        )
