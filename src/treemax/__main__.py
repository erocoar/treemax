"""Command-line interface."""
import click


@click.command()
@click.version_option()
def main() -> None:
    """treemax."""


if __name__ == "__main__":
    main(prog_name="treemax")  # pragma: no cover
