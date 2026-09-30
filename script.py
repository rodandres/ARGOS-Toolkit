from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src" / "argos"
DOCS_DIR = PROJECT_ROOT / "docs"
API_DIR = DOCS_DIR / "api"

EXCLUDED_PACKAGES = {
    "frames",
    "visualization",
}


def module_name(package_path: Path, file_path: Path) -> str:
    """Return the Python import path for a module."""
    relative = file_path.relative_to(SRC_DIR).with_suffix("")
    return "argos." + ".".join(relative.parts)


def package_name(package_path: Path) -> str:
    """Return the Python import path for a package."""
    relative = package_path.relative_to(SRC_DIR)
    return "argos." + ".".join(relative.parts)


def display_name(name: str) -> str:
    """Convert a Python name into a documentation title."""
    return name.replace("_", " ").replace("-", " ").title()


def generate_package_page(package_path: Path) -> tuple[str, str]:
    """Generate a Markdown API page for a package."""
    package = package_name(package_path)
    filename = f"{package_path.name}.md"

    modules = sorted(
        path
        for path in package_path.glob("*.py")
        if path.name != "__init__.py"
    )

    lines = [
        f"# {display_name(package_path.name)}",
        "",
    ]

    for module in modules:
        module_import = module_name(package_path, module)
        title = display_name(module.stem)

        lines.extend(
            [
                f"## {title}",
                "",
                f"::: {module_import}",
                "",
            ]
        )

    (API_DIR / filename).write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    return display_name(package_path.name), f"api/{filename}"


def generate_api_index() -> None:
    """Generate the API overview page."""
    content = """# API Reference

The API reference is generated automatically from the ARGOS source code.

It documents the public classes and functions exposed by the toolkit.
"""

    (API_DIR / "index.md").write_text(
        content,
        encoding="utf-8",
    )


def discover_packages() -> list[Path]:
    """Discover documentation packages under src/argos."""
    packages = []

    for path in sorted(SRC_DIR.iterdir()):
        if not path.is_dir():
            continue

        if path.name in EXCLUDED_PACKAGES:
            continue

        if (path / "__init__.py").exists():
            packages.append(path)

    return packages


def generate_mkdocs_config(packages: list[tuple[str, str]]) -> None:
    """Generate the MkDocs configuration."""
    lines = [
        "site_name: ARGOS Toolkit",
        "site_description: Aerospace Research for GNC Operational Scenarios toolkit",
        "site_url: https://rodandres.github.io/ARGOS-Toolkit/",
        "",
        "theme:",
        "  name: material",
        "",
        "plugins:",
        "  - search",
        "  - mkdocstrings:",
        "      handlers:",
        "        python:",
        "          paths:",
        "            - src",
        "",
        "nav:",
        "  - Home: index.md",
        "  - API Reference:",
        "      - Overview: api/index.md",
    ]

    for title, path in packages:
        lines.append(f"      - {title}: {path}")

    lines.append("")

    (PROJECT_ROOT / "mkdocs.yml").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def generate_home_page() -> None:
    """Generate the documentation home page."""
    content = """# ARGOS Toolkit

Documentation for the ARGOS Toolkit.

## API Reference

The [API Reference](api/index.md) contains automatically generated
documentation from the ARGOS source code.
"""

    (DOCS_DIR / "index.md").write_text(
        content,
        encoding="utf-8",
    )


def main() -> None:
    """Generate the complete MkDocs documentation structure."""
    API_DIR.mkdir(parents=True, exist_ok=True)

    generate_home_page()
    generate_api_index()

    packages = []

    for package in discover_packages():
        packages.append(generate_package_page(package))

    generate_mkdocs_config(packages)

    print("Documentation structure generated successfully.")
    print()
    print("Packages:")
    for title, path in packages:
        print(f"  - {title}: {path}")


if __name__ == "__main__":
    main()