def pytest_collection_modifyitems(session, config, items):
    """Guard: community tests must never import from enterprise.

    Note: This AST-based check catches static imports only. Dynamic imports
    via importlib.import_module() or __import__() are not detected here.
    The audit script (scripts/audit_public.sh) provides additional coverage.
    """
    import ast
    from pathlib import Path

    for item in items:
        fpath = Path(item.fspath)
        # Only skip the enterprise_clawloop directory itself
        if str(fpath).startswith("enterprise_clawloop/") or "/enterprise_clawloop/" in str(fpath):
            continue
        try:
            tree = ast.parse(fpath.read_text())
        except Exception:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and (
                node.module.startswith("enterprise")
            ):
                raise ValueError(
                    f"BOUNDARY VIOLATION: {fpath} imports from enterprise "
                    f"(line {node.lineno}). Community tests must NEVER "
                    f"import from enterprise/."
                )
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("enterprise"):
                        raise ValueError(
                            f"BOUNDARY VIOLATION: {fpath} imports from enterprise "
                            f"(line {node.lineno}). Community tests must NEVER "
                            f"import from enterprise/."
                        )
