#!/usr/bin/env python3
"""
CLI tool for managing configuration files.
"""

import os

import click
import yaml
from rich.console import Console
from rich.syntax import Syntax
from rich.table import Table

from ..utils import (
    create_default_config,
    find_config_file,
    load_config_from_file,
    save_config_to_file,
    validate_config,
)

console = Console()


@click.group()
def main():
    """Manage sensitive-term-guard configuration."""
    pass


@main.command()
@click.option("--output", "-o", default="config.yml", help="Output configuration file")
@click.option("--force", is_flag=True, help="Overwrite existing file")
def init(output, force):
    """Initialize a new configuration file with default values."""

    if os.path.exists(output) and not force:
        console.print(
            f"[red]Configuration file {output} already exists. Use --force to overwrite.[/red]"
        )
        return

    config = create_default_config()
    save_config_to_file(config, output)
    console.print(f"Created default configuration file: {output}[/green]")

    # Display the created config
    console.print("\n[bold]Generated configuration:[/bold]")
    with open(output, "r") as f:
        syntax = Syntax(f.read(), "yaml", theme="monokai", line_numbers=True)
        console.print(syntax)


@main.command()
@click.argument("config_file", type=click.Path(exists=True), required=False)
def validate(config_file):
    """Validate a configuration file."""

    if not config_file:
        config_file = find_config_file()
        if not config_file:
            console.print(
                "[red]No configuration file found. Specify path or run in directory with config file.[/red]"
            )
            return

    console.print(f"[blue]Validating configuration file: {config_file}[/blue]")

    try:
        config_dict = load_config_from_file(config_file)
        errors = validate_config(config_dict)

        if errors:
            console.print(f"[red]Validation failed with {len(errors)} errors:[/red]")
            for error in errors:
                console.print(f"  • {error}")
        else:
            console.print("[green]Configuration file is valid[/green]")

            # Show summary
            console.print("\n[bold]Configuration summary:[/bold]")
            display_config_summary(config_dict)

    except Exception as e:
        console.print(f"[red]Error validating configuration: {e}[/red]")


@main.command()
@click.argument("config_file", type=click.Path(exists=True), required=False)
def show(config_file):
    """Show current configuration settings."""

    if not config_file:
        config_file = find_config_file()
        if not config_file:
            console.print(
                "[red]No configuration file found. Specify path or run in directory with config file.[/red]"
            )
            return

    console.print(f"[blue]Configuration file: {config_file}[/blue]")

    try:
        config_dict = load_config_from_file(config_file)

        # Display as formatted YAML
        console.print("\n[bold]Configuration content:[/bold]")
        yaml_content = yaml.dump(config_dict, default_flow_style=False, indent=2)
        syntax = Syntax(yaml_content, "yaml", theme="monokai", line_numbers=True)
        console.print(syntax)

        # Show summary
        console.print("\n[bold]Summary:[/bold]")
        display_config_summary(config_dict)

    except Exception as e:
        console.print(f"[red]Error loading configuration: {e}[/red]")


@main.command()
@click.argument("key")
@click.argument("value")
@click.argument("config_file", type=click.Path(exists=True), required=False)
def set(key, value, config_file):
    """Set a configuration value."""

    if not config_file:
        config_file = find_config_file()
        if not config_file:
            console.print(
                "[red]No configuration file found. Specify path or run in directory with config file.[/red]"
            )
            return

    try:
        config_dict = load_config_from_file(config_file)

        # Parse the key path (e.g., "extraction.min_score")
        keys = key.split(".")
        current = config_dict

        # Navigate to the parent of the target key
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]

        # Try to convert value to appropriate type
        try:
            if value.lower() in ["true", "false"]:
                value = value.lower() == "true"
            elif "." in value:
                value = float(value)
            elif value.isdigit():
                value = int(value)
        except ValueError:
            pass  # Keep as string

        # Set the value
        current[keys[-1]] = value

        # Save the updated config
        save_config_to_file(config_dict, config_file)
        console.print(f"[green]✓ Set {key} = {value}[/green]")

    except Exception as e:
        console.print(f"[red]Error setting configuration value: {e}[/red]")


def display_config_summary(config_dict):
    """Display a summary of configuration settings."""
    table = Table(title="Configuration Summary")
    table.add_column("Section", style="cyan")
    table.add_column("Setting", style="yellow")
    table.add_column("Value", style="green")

    def add_section_rows(section_name, section_dict, prefix=""):
        for key, value in section_dict.items():
            if isinstance(value, dict):
                add_section_rows(section_name, value, f"{prefix}{key}.")
            else:
                table.add_row(
                    section_name if not prefix else "", f"{prefix}{key}", str(value)
                )

    for section, settings in config_dict.items():
        if isinstance(settings, dict):
            add_section_rows(section.title(), settings)
        else:
            table.add_row(section.title(), "", str(settings))

    console.print(table)


if __name__ == "__main__":
    main()
